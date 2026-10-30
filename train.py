# python libraries
import argparse
import os
import time

# torch
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, SequentialSampler

# our code
from libs.core import cfg
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_multimodal_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, save_checkpoint,
                        make_optimizer, make_scheduler, fix_random_seed, ModelEma, RunRecord)


def setup_distributed():
    """torchrun sets RANK / WORLD_SIZE / LOCAL_RANK and starts one process per GPU;
    a plain `python train.py` is one process on one GPU (or the CPU)."""
    if 'RANK' not in os.environ:
        return 0, 1, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank = int(os.environ['LOCAL_RANK'])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl')
        device = torch.device('cuda', local_rank)
    else:
        dist.init_process_group('gloo')
        device = torch.device('cpu')
    return dist.get_rank(), dist.get_world_size(), device


def main(args):
    """main function: train, then evaluate after every epoch"""

    """1. parameters / folders"""
    # ======================================= LOAD CONFIG =======================================================
    rank, world_size, device = setup_distributed()
    is_main = rank == 0   # only process 0 evaluates, saves checkpoints and prints
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)
    ckpt_folder = cfg['output_folder']
    if not cfg['dataset']['omni_emb_file']:
        cfg['model']['omni_head_dim'] = 0   # no OmniRetriever caption vectors: no omni head

    """2. datasets / dataloaders"""
    # ======================================= LOADER =======================================================
    data_cfg = dict(cfg['dataset'], scale_factor=cfg['model']['scale_factor'],
                    regression_range=cfg['model']['regression_range'], backbone_arch=cfg['model']['backbone_arch'])
    train_dataset = make_dataset('htstep', True, ['training'], **data_cfg)
    val_dataset = make_dataset('htstep', False, ['validation'], **data_cfg)
    # every process trains on its own share of the training videos, reshuffled every epoch
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
                                       seed=cfg['init_rand_seed'], drop_last=True)
    train_loader = make_data_loader(train_dataset, True, rng_generator, cfg['batch_size'], cfg['num_workers'],
                                    train_sampler)
    val_loader = None
    if is_main:
        val_loader = make_data_loader(val_dataset, False, None, cfg['batch_size'], cfg['num_workers'],
                                      SequentialSampler(val_dataset))

    # print('Data: %d train / %d val videos, %d unique train captions'
    #       % (len(train_dataset), len(val_dataset), len(train_dataset.pool_text)))

    """3. model, optimizer and scheduler"""
    # ======================================= MODEL CALLED HERE =======================================================
    model = make_multimodal_meta_arch('EventCaptionTransformer', **cfg['model'])
    model.to(device)
    # the caption pool only comes from the TRAIN split, validation captions are never used
    omni = train_dataset.pool_omni_emb
    model.set_caption_pool(
        torch.from_numpy(train_dataset.pool_emb).to(device), # (N, 512)
        torch.from_numpy(omni).to(device) if omni is not None else None # (N, 3584)
    )
    # print('Parameters: %.1fM' % (sum(p.numel() for p in model.parameters()) / 1e6))

    if args.eval:   # score a checkpoint, no training
        if is_main:
            model.load_state_dict(torch.load(args.eval, map_location=device)['state_dict'])
            m = valid_one_epoch(val_loader, model, train_dataset.pool_text, device)
            print(f"{args.eval}: R@0.5 {m['R@0.5']:.2f}  CIDEr {m['CIDEr']:.2f}", flush=True)
        if world_size > 1:
            dist.destroy_process_group()
        return

    # DDP copies the weights of process 0 to every process, and averages the gradients over the processes
    net = model
    if world_size > 1:
        net = DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None)
    optimizer = make_optimizer(model, cfg['opt'])
    scheduler = make_scheduler(optimizer, cfg['opt'], len(train_loader))
    model_ema = ModelEma(model, cfg['train_cfg']['ema_decay'])
    model_ema.module.set_caption_pool(model.pool_raw, model.omni_pool_raw )
    record = RunRecord(cfg, world_size) if is_main else None

    """4. train / evaluate loop"""
    max_epochs = cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    best = {'R@0.5': (-1.0, 0), 'CIDEr': (-1.0, 0)}   # metric -> (best value, its epoch)
    ckpt_path = os.path.join(ckpt_folder, 'model_best.pth.tar')
    for epoch in range(max_epochs):
        start = time.time()
        train_sampler.set_epoch(epoch)
        losses = train_one_epoch(train_loader, net, optimizer, scheduler, model_ema,
                                 cfg['train_cfg']['clip_grad_l2norm'], device)
        stop = False
        if is_main:
            m = valid_one_epoch(val_loader, model_ema.module, train_dataset.pool_text, device)
            mark = ''
            if m['CIDEr'] > best['CIDEr'][0]:
                best['CIDEr'] = (m['CIDEr'], epoch)
            if m['R@0.5'] > best['R@0.5'][0]:
                best['R@0.5'] = (m['R@0.5'], epoch)
                mark = '  <- best R@0.5, saved'
                save_checkpoint({'epoch': epoch + 1, 'state_dict': model_ema.module.state_dict(), 'metrics': m},
                                ckpt_folder, 'model_best.pth.tar')
            print(f"epoch {epoch + 1:2d}/{max_epochs}  {time.time() - start:4.0f}s  loss {losses['final_loss']:.3f}"
                  f" | R@0.5 {m['R@0.5']:.2f}  CIDEr {m['CIDEr']:.2f}{mark}", flush=True)
            record.add_epoch(epoch + 1, time.time() - start, losses, m,
                             {k: {'value': v, 'epoch': e + 1} for k, (v, e) in best.items()})
            # early stop: neither the best R@0.5 nor the best CIDEr has improved for `patience` epochs
            stop = args.patience > 0 and epoch - max(e for _, e in best.values()) >= args.patience
        if world_size > 1:   # every process follows the decision of process 0
            flag = [stop]
            dist.broadcast_object_list(flag, src=0)
            stop = flag[0]
        if stop:
            if is_main:
                print(f'Early stop: no new best R@0.5 / CIDEr for {args.patience} epochs', flush=True)
            break

    if is_main:
        print(f"Best R@0.5 = {best['R@0.5'][0]:.2f} -> {ckpt_path}")
        record.finish(stopped_early=stop, ckpt=ckpt_path)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train on HT-Step; parameters are in libs/core/config.py. '
                                            'Several GPUs: torchrun --nproc_per_node=N train.py')
    p.add_argument('--eval', default='', help='only score this checkpoint, no training')
    p.add_argument('--patience', type=int, default=3,
                   help='stop when neither R@0.5 nor CIDEr has improved for this many epochs (0 = never)')
    main(p.parse_args())
