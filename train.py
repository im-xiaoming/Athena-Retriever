"""Train the event segmentation + captioning model on YouCook2, then score it on the validation set.

  python train.py

Settings are in config.py. The best R@0.5 checkpoint is saved to OUT_DIR/best.pth.tar.
"""
import copy
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from pycocoevalcap.cider.cider import Cider
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import config as C
from dataset import YouCook2Dataset, trivial_batch_collator, worker_init_reset_seed
from model import EventCaptionModel


def iou(a, b):
    """IoU of two segments (start, end)."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def mbr_pick(s, pool_n, scale, k=20):
    """Pick a caption by consensus instead of taking the top-scoring one.

    s (N,) scores every caption in the pool. Take the top k, turn their scores into
    probabilities, and return the candidate most similar to the whole group
    (minimum Bayes risk). A caption backed by many strong candidates beats one that
    is on top by chance.
    """
    top = s.topk(k)
    p = F.softmax(scale * top.values, dim=0)
    cand = pool_n[top.indices]
    agree = (cand @ cand.t()) @ p
    return int(top.indices[agree.argmax()])


@torch.no_grad()
def evaluate(model, loader, pool_text):
    """Segmentation and captioning scores on the validation set.

    R@t      : share of GT steps matched by one of the top-k predictions (k = number of GT steps) with IoU >= t
    mIoU     : mean best IoU of every GT step
    txt_sim  : cosine (InternVideo2 text space) between the caption picked for a matched step and its GT caption
    CIDEr    : word overlap of the same picks
    G R1@0.5 : grounding, each GT caption as the query must find its own span with IoU >= 0.5
    Captioning is scored only on GT steps matched by a prediction with IoU >= 0.3.
    """
    model.eval()
    dev = model.points.device
    pool_f = model.embed_head.embed_captions(model.pool_raw)
    pool_n = F.normalize(model.pool_raw, dim=-1)
    scale = float(model.embed_head.logit_scale.exp().clamp(max=100))

    best_ious, txt_sim, g_ious = [], [], []
    gts_txt, hyp = {}, {}
    for batch in loader:
        for r, x in zip(model(batch), batch):
            gts = r['gt_segments']
            g_ious += [iou(p, g) for p, g in zip(r['ground_segments'].tolist(), gts)]
            # keep as many predictions (best score first) as the video has GT steps
            k = min(len(gts), len(r['segments']))
            preds = r['segments'][:k].tolist()
            cap_n = F.normalize(x['cap_emb'].to(dev), dim=-1)
            for gi, gt in enumerate(gts):
                vals = [iou(p, gt) for p in preds] or [0.0]
                j = int(np.argmax(vals))   # the prediction that matches this GT step best
                best_ious.append(vals[j])
                if vals[j] < 0.3:          # captioning is scored on matched steps only
                    continue
                i = mbr_pick(pool_f @ r['embeds'][j].to(dev), pool_n, scale)
                txt_sim.append(float(pool_n[i] @ cap_n[gi]))
                gts_txt[len(gts_txt)] = [x['cap_text'][gi]]
                hyp[len(hyp)] = [pool_text[i]]
    model.train()

    best_ious = np.array(best_ious)
    out = {'R@0.3': 100 * np.mean(best_ious >= 0.3), 'R@0.5': 100 * np.mean(best_ious >= 0.5),
           'R@0.7': 100 * np.mean(best_ious >= 0.7), 'mIoU': 100 * np.mean(best_ious),
           'txt_sim': np.mean(txt_sim), 'CIDEr': Cider().compute_score(gts_txt, hyp)[0] * 100,
           'G R1@0.5': 100 * np.mean(np.array(g_ious) >= 0.5)}
    return out


def build_optimizer(model):
    """AdamW with three parameter groups (as ActionFormer, plus one for the caption projection):
      decay    : weights of Linear / Conv layers, with weight decay
      no_decay : biases, LayerNorm, learnable scales and other parameters, no weight decay
      clip     : clip_proj (caption vectors -> event space), its own learning rate CLIP_PROJ_LR
    """
    decay, no_decay, clip = [], [], []
    for mod_name, mod in model.named_modules():
        for par_name, p in mod.named_parameters(recurse=False):
            if 'clip_proj' in mod_name:
                clip.append(p)
            elif par_name == 'weight' and isinstance(mod, (torch.nn.Linear, torch.nn.Conv1d)):
                decay.append(p)
            else:
                no_decay.append(p)
    return torch.optim.AdamW([
        {'params': decay, 'weight_decay': C.WEIGHT_DECAY},
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': clip, 'weight_decay': 0.0, 'lr': C.CLIP_PROJ_LR},
    ], lr=C.LR)


def lr_factor(step, warmup_steps, total_steps):
    """Linear warm-up from 0, then cosine decay to 0; the learning rate is updated every iteration."""
    if step < warmup_steps:
        return step / (warmup_steps - 1)
    return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))


class ModelEma:
    """Exponential moving average of the weights, as ActionFormer does: evaluation and the saved
    checkpoint use the averaged weights."""

    def __init__(self, model):
        self.module = copy.deepcopy(model)
        self.updates = 0
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        # decay warm-up: a fixed 0.999 would still hold most of the random init after the first epoch
        self.updates += 1
        d = min(C.EMA_DECAY, (1 + self.updates) / (10 + self.updates))
        for e, m in zip(self.module.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)


def main():
    # ======================================= SEED =======================================================
    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)
    dev = torch.device('cuda')

    # ======================================= LOADER =======================================================
    train_set = YouCook2Dataset('training', is_training=True)
    val_set = YouCook2Dataset('validation', is_training=False)
    train_loader = DataLoader(train_set, batch_size=C.BATCH_SIZE, num_workers=C.NUM_WORKERS,
                              sampler=RandomSampler(train_set), collate_fn=trivial_batch_collator,
                              worker_init_fn=worker_init_reset_seed, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=C.BATCH_SIZE, num_workers=C.NUM_WORKERS,
                            sampler=SequentialSampler(val_set), collate_fn=trivial_batch_collator)
    pool_text = train_set.pool_text
    print('Data: %d train / %d val videos, %d unique train captions'
          % (len(train_set), len(val_set), len(pool_text)))

    # ======================================= MODEL CALLED HERE =======================================================
    model = EventCaptionModel(torch.from_numpy(train_set.pool_emb)).to(dev)
    print('Parameters: %.1fM' % (sum(p.numel() for p in model.parameters()) / 1e6))

    opt = build_optimizer(model)
    n_epochs = C.EPOCHS + C.WARMUP_EPOCHS
    warmup_steps = C.WARMUP_EPOCHS * len(train_loader)
    total_steps = n_epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, warmup_steps, total_steps))
    ema = ModelEma(model)
    os.makedirs(C.OUT_DIR, exist_ok=True)

    # ======================================= TRAIN =======================================================
    best = -1
    for ep in range(n_epochs):
        t0 = time.time()
        losses = {}
        for batch in train_loader:
            L = model(batch)
            opt.zero_grad()
            L['final_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.CLIP_GRAD)
            opt.step()
            sched.step()
            ema.update(model)
            for k, v in L.items():
                losses[k] = losses.get(k, 0.0) + v.item() / len(train_loader)

        m = evaluate(ema.module, val_loader, pool_text)
        mark = ''
        if m['R@0.5'] > best:
            best = m['R@0.5']
            mark = '  <- best R@0.5, saved'
            torch.save({'epoch': ep + 1, 'state_dict': ema.module.state_dict(), 'metrics': m},
                       os.path.join(C.OUT_DIR, 'best.pth.tar'))
        print('epoch %2d/%d  %4.0fs  loss %.3f | R@0.3 %.2f  R@0.5 %.2f  R@0.7 %.2f  mIoU %.2f | '
              'txt_sim %.3f  CIDEr %.2f | G R1@0.5 %.2f%s'
              % (ep + 1, n_epochs, time.time() - t0, losses['final_loss'], m['R@0.3'], m['R@0.5'], m['R@0.7'],
                 m['mIoU'], m['txt_sim'], m['CIDEr'], m['G R1@0.5'], mark), flush=True)
    print('Best R@0.5 = %.2f' % best)


if __name__ == '__main__':
    main()
