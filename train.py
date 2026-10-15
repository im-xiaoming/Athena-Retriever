# thư viện python
import os
import time

# thư viện torch
import torch
from torch.utils.data import RandomSampler, SequentialSampler

# code của mình
from libs.core import cfg
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_multimodal_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, save_checkpoint,
                        make_optimizer, make_scheduler, fix_random_seed, ModelEma)


def main():
    """hàm chính: train rồi đánh giá sau mỗi epoch"""

    """1. thiết lập tham số / thư mục"""
    # ======================================= LOAD CONFIG =======================================================
    device = torch.device('cuda')
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)
    ckpt_folder = cfg['output_folder']

    """2. tạo dataset / dataloader"""
    # ======================================= LOADER =======================================================
    data_cfg = dict(cfg['dataset'], scale_factor=cfg['model']['scale_factor'],
                    regression_range=cfg['model']['regression_range'], backbone_arch=cfg['model']['backbone_arch'])
    train_dataset = make_dataset('youcook2', True, ['training'], **data_cfg)
    val_dataset = make_dataset('youcook2', False, ['validation'], **data_cfg)
    train_loader = make_data_loader(train_dataset, True, rng_generator, cfg['batch_size'], cfg['num_workers'],
                                    RandomSampler(train_dataset, generator=rng_generator))
    val_loader = make_data_loader(val_dataset, False, None, cfg['batch_size'], cfg['num_workers'],
                                  SequentialSampler(val_dataset))
    print('Data: %d train / %d val videos, %d unique train captions'
          % (len(train_dataset), len(val_dataset), len(train_dataset.pool_text)))

    """3. tạo model, optimizer và scheduler"""
    # ======================================= MODEL CALLED HERE =======================================================
    model = make_multimodal_meta_arch('EventCaptionTransformer', **cfg['model'])
    model.to(device)
    # pool caption chỉ lấy từ tập TRAIN, caption của tập val không bao giờ được dùng
    model.set_caption_pool(torch.from_numpy(train_dataset.pool_emb).to(device))
    print('Parameters: %.1fM' % (sum(p.numel() for p in model.parameters()) / 1e6))

    optimizer = make_optimizer(model, cfg['opt'])
    scheduler = make_scheduler(optimizer, cfg['opt'], len(train_loader))
    model_ema = ModelEma(model, cfg['train_cfg']['ema_decay'])
    model_ema.module.set_caption_pool(model.pool_raw)

    """4. vòng lặp train / đánh giá"""
    max_epochs = cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    best_r05 = -1.0
    for epoch in range(max_epochs):
        start = time.time()
        losses = train_one_epoch(train_loader, model, optimizer, scheduler, model_ema,
                                 cfg['train_cfg']['clip_grad_l2norm'], device)
        m = valid_one_epoch(val_loader, model_ema.module, train_dataset.pool_text, device)

        mark = ''
        if m['R@0.5'] > best_r05:
            best_r05 = m['R@0.5']
            mark = '  <- best R@0.5, saved'
            save_checkpoint({'epoch': epoch + 1, 'state_dict': model_ema.module.state_dict(), 'metrics': m},
                            ckpt_folder, 'model_best.pth.tar')
        print('epoch %2d/%d  %4.0fs  loss %.3f | R@0.3 %.2f  R@0.5 %.2f  R@0.7 %.2f  mIoU %.2f | '
              'txt_sim %.3f  CIDEr %.2f | G R1@0.5 %.2f%s'
              % (epoch + 1, max_epochs, time.time() - start, losses['final_loss'], m['R@0.3'], m['R@0.5'],
                 m['R@0.7'], m['mIoU'], m['txt_sim'], m['CIDEr'], m['G R1@0.5'], mark), flush=True)
    print('Best R@0.5 = %.2f -> %s' % (best_r05, os.path.join(ckpt_folder, 'model_best.pth.tar')))


if __name__ == '__main__':
    main()
