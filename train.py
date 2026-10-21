"""Train the event segmentation + captioning model on HT-Step.

  python train.py configs/htstep_event.yaml --output run1
  python train.py configs/htstep_event.yaml --output run2 --set init_rand_seed=2
  python train.py configs/htstep_event.yaml --eval ckpt/run1/best_cap.pth.tar

Every training run writes experiments/runs/<host>-<output>.json with the git commit,
the resolved config, the overrides, per-epoch metrics and a full evaluation of the
best captioning checkpoint. tools/summarize_runs.py turns those files into a table.
"""
import argparse
import os
import time

import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from libs.datasets import StepCaptionDataset, trivial_batch_collator, worker_init_reset_seed
from libs.modeling import EventCaptionTransformer, upgrade_state_dict
from libs.utils import make_scheduler, fix_random_seed
from libs.utils import log_table as table
from libs.utils.evaluation import collect_predictions, dump_predictions, evaluate
from libs.utils.run_record import RunRecord, apply_overrides
from libs.utils.seg_metrics import best_threshold
from libs.utils.train_utils import ModelEma, build_optimizer, load_ckpt, load_weights


def val_loader(cfg, d):
    return DataLoader(d, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                      sampler=SequentialSampler(d), collate_fn=trivial_batch_collator)


def build_loaders(cfg, rng):
    """HT-Step train / val datasets and loaders: ds (train), dv, dl (train loader), dlv."""
    ds = StepCaptionDataset(True, cfg['train_split'], **cfg['dataset'])
    dv = StepCaptionDataset(False, cfg['val_split'], **cfg['dataset'])
    dl = DataLoader(ds, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                    sampler=RandomSampler(ds), collate_fn=trivial_batch_collator,
                    worker_init_fn=worker_init_reset_seed, drop_last=True, generator=rng,
                    persistent_workers=cfg['num_workers'] > 0)
    return ds, dv, dl, val_loader(cfg, dv)


def eval_only(a, model, dlv, dev, pool_raw, pool_text, cfg):
    """Score one checkpoint under several NMS / ranking settings, with all variants. --thr-from-train: the
    densevid score threshold is chosen on the training videos."""
    ck = load_weights(model, a.eval)
    print('Checkpoint   : %s (epoch %d)' % (a.eval, ck['epoch'] + 1), flush=True)
    if a.dump_preds:   # with the default NMS / iou_power only
        return dump_predictions(model, dlv, dev, a.dump_preds)
    powers = a.iou_power.split(',') if a.iou_power else [str(model.iou_power)]
    for power in powers:
        model.iou_power = float(power)
        for spec in a.nms.split(','):
            kind, *rest = spec.split(':')
            model.nms_cfg = {'soft': kind == 'soft', 'iou': float(rest[0]),
                             'sigma': float(rest[1]) if len(rest) > 1 else 0.5}
            thr = None
            if a.thr_from_train:
                train = val_loader(cfg, StepCaptionDataset(False, cfg['train_split'], **cfg['dataset']))
                thr = best_threshold([collect_predictions(model, train, dev)])
                print('Threshold    : %.2f, best densevid F1 over the training videos' % thr, flush=True)
            m = evaluate(model, dlv, dev, pool_raw, pool_text, full=not a.fast, std=True, thr=thr)
            print('\niou_power %s  NMS %-14s' % (power, spec)
                  + '  '.join('%s %.3f' % (k, v) for k, v in m.items()), flush=True)


def main(a):
    # ======================================= LOAD CONFIG =======================================================
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    apply_overrides(cfg, a.set)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = fix_random_seed(cfg.get('init_rand_seed', 1234567891), include_cuda=True)
    out_dir = os.path.join(cfg['output_folder'], a.output)


    # ======================================= LOADER =======================================================
    ds, dv, dl, dlv = build_loaders(cfg, rng) # dataset_train, dataset_val, loader_train, loader_val
    print('Data         : %d train / %d val videos' % (len(ds), len(dv)), flush=True)
    # the input width follows the feature source; stored in the run record's config
    cfg['model']['input_dim_V'], cfg['model']['input_dim_A'] = ds.feat_dims
    print('Features     : InternVideo2 %s + BEATs a768, visual %d + audio %d channels' % (
        '+'.join(ds.iv2_video_keys), *ds.feat_dims), flush=True)
    cfg['model']['clip_dim'] = ds.pool_emb.shape[1]   # caption space
    if cfg['model']['train_cfg'].get('loss_weight_ground', 0) > 0:
        table.use_ground_columns()


    # ======================================= MODEL CALLED HERE =======================================================
    model = EventCaptionTransformer(**cfg['model']).to(dev)
    print('Parameters   : %.1fM' % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)

    # the caption pool comes from the TRAIN split only; val captions are never used
    pool_raw = torch.from_numpy(ds.pool_emb).to(dev)
    model.set_caption_pool(pool_raw)
    print('Caption pool : %d unique train captions' % len(ds.pool_text), flush=True)
    if a.init:   # fine-tune: start from these weights (new modules keep their init)
        sd = upgrade_state_dict(load_ckpt(a.init)['state_dict'])
        miss, unused = model.load_state_dict(sd, strict=False)
        print('Init         : %s (%d missing, %d unused keys%s)' % (
            a.init, len(miss), len(unused), ': ' + ', '.join(sorted({k.split('.')[0] for k in miss + unused}))
            if miss or unused else ''), flush=True)
    if a.eval:
        return eval_only(a, model, dlv, dev, pool_raw, ds.pool_text, cfg)
    if a.finalize:   # redo the final evaluation of a finished run, e.g. after a crash in it
        rec = RunRecord.load(a.output)
        return finalize(rec, model, dlv, dev, pool_raw, ds.pool_text)
    os.makedirs(out_dir, exist_ok=True)
    rec = RunRecord(a, cfg, out_dir)
    print('Output       : %s' % out_dir)
    print('Run record   : %s (commit %s%s)' % (rec.path, rec.data['git']['commit'],
                                               ', DIRTY' if rec.data['git']['dirty'] else ''), flush=True)

    opt = build_optimizer(model, cfg['opt'])
    sch = make_scheduler(opt, cfg['opt'], len(dl), a.epochs)
    ema = ModelEma(model, cfg['train_cfg']['ema_decay']) if cfg['train_cfg'].get('ema_decay', 0) > 0 else None
    if ema is not None:
        print('EMA         : decay %g, evaluated and saved instead of the raw weights' % ema.decay, flush=True)
    net = ema.module if ema is not None else model   # the weights that are evaluated and saved
    n_ep = a.epochs + cfg['opt']['warmup_epochs']
    # metric -> (best value, epoch, checkpoint name): each metric keeps its own best checkpoint
    best = {'R@0.5': (-1.0, 0, 'best_seg'), 'CIDEr': (-1.0, 0, 'best_cap')}
    table.print_header()
    for ep in range(n_ep):
        t0 = time.time()
        acc = {}   # sum of every loss over the epoch
        for batch in dl:
            for x in batch:
                x['feats'] = {k: v.to(dev) for k, v in x['feats'].items()}
            L = model(batch)
            opt.zero_grad(set_to_none=True)
            L['final_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train_cfg']['clip_grad_l2norm'])
            opt.step()
            sch.step()   # the schedule steps every iteration, not every epoch
            if ema is not None:
                ema.update(model)
            for k, v in L.items():
                acc[k] = acc.get(k, 0.0) + v.detach().item()
        m = evaluate(net, dlv, dev, pool_raw, ds.pool_text, std=a.std_metrics)
        mark = ''
        for key, (val, _, name) in best.items():
            if m[key] > val:
                best[key] = (m[key], ep, name)
                mark += {'R@0.5': 'S', 'CIDEr': 'C'}[key]
                torch.save({'epoch': ep, 'state_dict': net.state_dict(), 'metrics': m},
                           os.path.join(out_dir, name + '.pth.tar'))
        vals = dict(m, **{k: v / len(dl) for k, v in acc.items()})   # val metrics + mean train losses
        table.print_row('%d/%d' % (ep + 1, n_ep), time.time() - t0, sch.get_last_lr()[0], vals, mark)
        rec.data['history'].append(dict(vals, epoch=ep + 1, seconds=round(time.time() - t0, 1)))
        rec.data['best'] = {k: {'value': v, 'epoch': e + 1, 'ckpt': n} for k, (v, e, n) in best.items()}
        rec.save()
        # early stop: no best (R@0.5, CIDEr) has improved for `patience` epochs
        if a.patience and ep - max(e for _, e, _ in best.values()) >= a.patience:
            print('Early stop   : no new best %s for %d epochs' % (' / '.join(best), a.patience), flush=True)
            rec.data['stopped_early'] = ep + 1
            rec.save()
            break
    print(table.RULE)
    for key, (val, ep, name) in best.items():
        print('Best %-6s = %6.2f at epoch %d -> %s' % (
            key, val, ep + 1, os.path.join(out_dir, name + '.pth.tar')), flush=True)

    finalize(rec, model, dlv, dev, pool_raw, ds.pool_text)


def finalize(rec, model, dlv, dev, pool_raw, pool_text):
    """Full evaluation of the run's best captioning checkpoint, stored in the run record."""
    load_weights(model, os.path.join(rec.data['out_dir'], 'best_cap.pth.tar'))
    final = evaluate(model, dlv, dev, pool_raw, pool_text, full=True, std=True)
    rec.data['final_eval'] = dict(final, ckpt='best_cap')
    rec.data['finished'] = rec.data.get('finished') or time.strftime('%Y-%m-%d %H:%M:%S')
    rec.save()
    print('Final eval   : ' + '  '.join('%s %.3f' % (k, v) for k, v in final.items()), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config'); p.add_argument('--output', default='ev1')
    p.add_argument('--epochs', type=int, default=10,
                   help='40 epochs overfit from epoch 7; 10 is enough')
    p.add_argument('--eval', default='', help='only score this checkpoint, no training')
    p.add_argument('--init', default='', help='start training from the weights of this checkpoint (fine-tune)')
    p.add_argument('--patience', type=int, default=0,
                   help='stop when neither best R@0.5 nor best CIDEr improved for this many epochs (0 = off)')
    p.add_argument('--nms', default='soft:0.7:0.5',
                   help='NMS settings to score with --eval, comma separated, e.g. soft:0.7:0.5,hard:0.5')
    p.add_argument('--iou-power', default='',
                   help='--eval: IoU-score weights to try, comma separated; empty keeps the config value')
    p.add_argument('--std-metrics', action='store_true',
                   help='also compute the standard metrics (mAP, densevid P/R/F1) every epoch; always on in '
                        '--eval and the final evaluation')
    p.add_argument('--dump-preds', default='',
                   help='--eval: only write the predictions and GT of the validation set to this folder (densevid_eval format)')
    p.add_argument('--thr-from-train', action='store_true',
                   help='--eval: choose the densevid score threshold on the training videos')
    p.add_argument('--fast', action='store_true', help='--eval: skip the slow variants')
    p.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                   help='override the config, e.g. --set init_rand_seed=2 model.train_cfg.loss_weight_span=0.5')
    p.add_argument('--note', default='', help='free text stored in the run record')
    p.add_argument('--finalize', action='store_true',
                   help='only redo the final evaluation of the finished run --output (same --set as training)')
    main(p.parse_args())
