import os
import copy
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from pycocoevalcap.cider.cider import Cider

from .lr_schedulers import LinearWarmupCosineAnnealingLR


def fix_random_seed(seed, include_cuda=True):
    """Cố định seed cho python, numpy, torch; nếu include_cuda thì cho cuDNN / cuBLAS chạy tất định.
    Trả về torch generator để DataLoader dùng."""
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if include_cuda:
        # tắt cudnn benchmark để chạy lại ra cùng kết quả
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # cần cho CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    return rng_generator


def save_checkpoint(state, file_folder, file_name):
    """lưu checkpoint ra file"""
    if not os.path.exists(file_folder):
        os.makedirs(file_folder)
    torch.save(state, os.path.join(file_folder, file_name))


def make_optimizer(model, optimizer_config):
    """AdamW với ba nhóm tham số (giống ActionFormer, thêm một nhóm cho phép chiếu caption):
      decay    : weight của các lớp Linear / Conv, có weight decay
      no_decay : bias, LayerNorm, các hệ số scale và tham số khác, không weight decay
      clip     : clip_proj (chiếu vector caption sang không gian event), learning rate riêng clip_proj_lr
    """
    decay, no_decay, clip = [], [], []
    for mod_name, mod in model.named_modules():
        for par_name, p in mod.named_parameters(recurse=False):
            if 'clip_proj' in mod_name:
                clip.append(p)
            elif par_name.endswith('weight') and isinstance(mod, (torch.nn.Linear, torch.nn.Conv1d)):
                decay.append(p)
            else:
                no_decay.append(p)
    groups = [
        {'params': decay, 'weight_decay': optimizer_config['weight_decay']},
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': clip, 'weight_decay': 0.0, 'lr': optimizer_config['clip_proj_lr']},
    ]
    return torch.optim.AdamW(groups, lr=optimizer_config['learning_rate'])


def make_scheduler(optimizer, optimizer_config, num_iters_per_epoch):
    """Warm-up tuyến tính trong warmup_epochs, sau đó giảm theo cosine; cập nhật sau mỗi iteration."""
    warmup_steps = optimizer_config['warmup_epochs'] * num_iters_per_epoch
    max_steps = (optimizer_config['epochs'] + optimizer_config['warmup_epochs']) * num_iters_per_epoch
    return LinearWarmupCosineAnnealingLR(optimizer, warmup_steps, max_steps)


class ModelEma(torch.nn.Module):
    """Trung bình trượt (EMA) của trọng số, giống ActionFormer: đánh giá và lưu checkpoint đều dùng
    trọng số trung bình này."""

    def __init__(self, model, decay=0.999):
        super().__init__()
        # copy model ra một bản để cộng dồn trung bình trượt
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        # decay tăng dần lúc đầu: nếu để cố định 0.999 thì sau epoch đầu trọng số vẫn gần như là lúc khởi tạo,
        # đánh giá mấy epoch đầu sẽ ra điểm của một model chưa học gì
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(d).add_(model_v.detach(), alpha=1 - d)
            else:
                ema_v.copy_(model_v)


def train_one_epoch(
    train_loader,      # DataLoader, mỗi batch là list các dict (xem YouCook2Dataset.__getitem__)
    model,
    optimizer,
    scheduler,
    model_ema,         # ModelEma, cập nhật sau mỗi bước
    clip_grad_l2norm,  # norm L2 tối đa của gradient
    device,
):
    """Train một epoch, trả về trung bình của từng loss."""
    model.train()
    losses = {}
    for video_list in train_loader:
        for x in video_list:
            x['feats'] = {k: v.to(device) for k, v in x['feats'].items()}
        losses_dict = model(video_list)
        optimizer.zero_grad(set_to_none=True)
        losses_dict['final_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_l2norm)
        optimizer.step()
        scheduler.step()   # learning rate cập nhật sau mỗi iteration, không phải mỗi epoch
        model_ema.update(model)
        for k, v in losses_dict.items():
            losses[k] = losses.get(k, 0.0) + v.item() / len(train_loader)
    return losses


def iou(a, b):
    """IoU của hai đoạn (đầu, cuối)."""
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    u = (a[1] - a[0]) + (b[1] - b[0]) - i
    return i / u if u > 0 else 0.0


def mbr_pick(
    s,       # (N,) điểm của từng caption trong pool cho một đoạn
    pool_n,  # (N, 512) vector caption của pool đã chuẩn hóa L2
    scale,   # nhiệt độ của softmax (logit scale của model)
    k=20,    # số caption điểm cao nhất được đem ra "bầu"
):
    """Chọn caption theo số đông thay vì lấy caption điểm cao nhất.

    Lấy k caption điểm cao nhất, đổi điểm thành xác suất, rồi chọn caption giống cả nhóm nhất
    (minimum Bayes risk). Caption được nhiều ứng viên mạnh ủng hộ sẽ thắng caption chỉ tình cờ đứng đầu.
    """
    top = s.topk(k)
    p = F.softmax(scale * top.values, dim=0)
    cand = pool_n[top.indices]
    agree = (cand @ cand.t()) @ p
    return int(top.indices[agree.argmax()])


@torch.no_grad()
def valid_one_epoch(
    val_loader,  # DataLoader của tập validation
    model,       # model EMA
    pool_text,   # [N] text của pool caption train, đúng thứ tự trong pool
    device,
):
    """Chấm điểm tách bước và gán caption trên tập validation.

    R@0.5 : tỉ lệ bước GT khớp với một trong top-k dự đoán (k = số bước GT của video) với IoU >= 0.5
    CIDEr : độ đo caption thông dụng, dựa trên từ trùng nhau; chỉ chấm trên các bước GT
            có dự đoán khớp với IoU >= 0.3
    """
    model.eval()
    pool_f = model.embed_head.embed_captions(model.pool_raw)
    pool_n = F.normalize(model.pool_raw, dim=-1)
    scale = float(model.embed_head.logit_scale.exp().clamp(max=100))

    best_ious = []
    gts_txt, hyp = {}, {}
    for video_list in val_loader:
        for x in video_list:
            x['feats'] = {k: v.to(device) for k, v in x['feats'].items()}
        for r, x in zip(model(video_list), video_list):
            gts = r['gt_segments']
            # giữ số dự đoán (điểm cao trước) bằng số bước GT của video
            k = min(len(gts), len(r['segments']))
            preds = r['segments'][:k].tolist()
            for i, gt in enumerate(gts):
                vals = [iou(p, gt) for p in preds] or [0.0]
                j = int(np.argmax(vals))   # dự đoán khớp với bước GT này nhất
                best_ious.append(vals[j])
                if vals[j] < 0.3:          # chỉ chấm caption ở các bước đã khớp
                    continue
                i = mbr_pick(pool_f @ r['embeds'][j].to(device), pool_n, scale)
                gts_txt[len(gts_txt)] = [x['cap_text'][i]]
                hyp[len(hyp)] = [pool_text[i]]

    return {'R@0.5': 100 * np.mean(np.array(best_ious) >= 0.5),
            'CIDEr': Cider().compute_score(gts_txt, hyp)[0] * 100}
