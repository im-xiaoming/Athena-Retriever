"""Train the event segmentation + captioning model on YouCook2.

  python train_event.py configs/youcook2_event.yaml --output run1
  python train_event.py configs/youcook2_event.yaml --output run2 --set init_rand_seed=2
  python train_event.py configs/youcook2_event.yaml --eval ckpt/run1/best_cap.pth.tar

Every training run writes experiments/runs/<host>-<output>.json with the git commit,
the resolved config, the overrides, per-epoch metrics and a full evaluation of the
best captioning checkpoint. tools/summarize_runs.py turns those files into a table.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from libs.datasets import YouCook2CaptionDataset, trivial_batch_collator, worker_init_reset_seed
from libs.modeling import EventCaptionTransformer, MaskedConv1D, upgrade_state_dict
from libs.utils import make_scheduler, fix_random_seed

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments', 'runs')


def build_optimizer(model, opt):
    decay, no_decay, clip = set(), set(), set()
    white = (torch.nn.Linear, torch.nn.Conv1d, MaskedConv1D)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            fpn = '%s.%s' % (mn, pn) if mn else pn
            if 'clip_proj' in fpn:
                clip.add(fpn)
            elif pn.endswith('weight') and isinstance(m, white):
                decay.add(fpn)
            else:
                no_decay.add(fpn)
    pd = dict(model.named_parameters())
    assert not (pd.keys() - (decay | no_decay | clip))
    return torch.optim.AdamW([
        {'params': [pd[n] for n in sorted(decay)], 'weight_decay': opt['weight_decay']},
        {'params': [pd[n] for n in sorted(no_decay)], 'weight_decay': 0.0},
        {'params': [pd[n] for n in sorted(clip)], 'weight_decay': 0.0, 'lr': opt['clip_proj_lr']},
    ], lr=opt['learning_rate'])


def load_ckpt(path):
    """torch.load for our own checkpoints on any torch version.

    torch >= 2.6 defaults to weights_only=True, which rejects the metrics dict stored
    next to the weights; torch 1.11 has no weights_only argument at all.
    """
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def load_weights(model, path):
    """Load a training checkpoint into the model (older layouts are upgraded); returns the checkpoint."""
    ck = load_ckpt(path)
    model.load_state_dict(upgrade_state_dict(ck['state_dict']))
    return ck


def iou(a, b):
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    u = (a[1] - a[0]) + (b[1] - b[0]) - i
    return i / u if u > 0 else 0.0


def mbr_pick(s, pool_n, scale, k=20):
    """Pick a caption by consensus instead of taking the top-scoring one.

    s (N,) scores every caption in the pool. Take the top k, turn their scores into
    probabilities, and return the candidate most similar to the whole group
    (minimum Bayes risk). A caption backed by many strong candidates beats one that
    is on top by chance. Returns (top-1 index, MBR index).
    """
    top = s.topk(k)
    p = F.softmax(scale * top.values, dim=0)
    cand = pool_n[top.indices]
    agree = (cand @ cand.t()) @ p
    return int(top.indices[0]), int(top.indices[agree.argmax()])


def meteor(gts, hyp, timeout=60):
    """METEOR via pycocoevalcap (java), in a separate Python process with a time limit.

    On Colab the java helper can die mid-call while pycocoevalcap holds its lock;
    the object's __del__ then waits on that lock forever. Isolating it in a child
    process means a failure or hang only costs this metric, never the evaluation.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, 'in.json')
        with open(src, 'w') as f:
            json.dump({'gts': {str(k): v for k, v in gts.items()}, 'hyp': {str(k): v for k, v in hyp.items()}}, f)
        code = ('import json,sys; from pycocoevalcap.meteor.meteor import Meteor; '
                'd=json.load(open(sys.argv[1])); print(Meteor().compute_score(d["gts"], d["hyp"])[0]*100, flush=True)')
        try:
            r = subprocess.run([sys.executable, '-c', code, src], capture_output=True, text=True, timeout=timeout)
            return float(r.stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            print('METEOR skipped: timed out after %ds' % timeout, flush=True)
        except (ValueError, IndexError):
            print('METEOR skipped: %s' % (r.stderr.strip().splitlines() or ['no output'])[-1][:200], flush=True)
    return None


@torch.no_grad()
def evaluate(model, loader, device, pool_raw, pool_text, ths=(0.3, 0.5, 0.7), full=False):
    """Score segmentation and captioning on the validation set.

    Captioning is scored only on GT segments matched by a prediction with IoU >= 0.3.
    cos_gt   : cosine between the predicted segment vector and its GT caption
    cos_rand : the same against a random caption, i.e. the floor
    ret_sim  : how close the chosen caption is to the GT caption, in ONE-PEACE space
    CIDEr    : standard captioning metric, word overlap, independent of ONE-PEACE
    full     : also score variants (top-1, single-point vector, each embedding space
               separately when the OmniRetriever teacher is on, oracle GT spans)
    With the teacher, a caption's score is the sum of cosines in both spaces.
    """
    from pycocoevalcap.cider.cider import Cider
    model.eval()
    pool_f = F.normalize(model.embed_head.clip_proj(pool_raw), dim=-1)
    # metrics and the consensus pick always use ONE-PEACE vectors, whatever space the model is trained in
    pool_n = F.normalize(getattr(model, 'pool_metric', None) if getattr(model, 'pool_metric', None) is not None
                         else pool_raw, dim=-1)
    same_space = pool_raw.shape[1] == model.embed_head.clip_proj.in_features and getattr(model, 'pool_metric', None) is None
    scale = float(model.embed_head.logit_scale.exp().clamp(max=100))
    omni = model.omni_dim > 0
    if omni:
        o_txt = model.omni_text.masked_fill(~model.omni_text_ok[:, None], 0)
    cov = {t: 0 for t in ths}
    n_gt = n_pred = 0
    best_ious, cos_gt, cos_rand = [], [], []
    ret = {v: [] for v in ('top1', 'mbr', 'pt_top1', 'mbr_onepeace', 'mbr_omni', 'oracle')}
    gts_txt, hyp = {}, {v: {} for v in ret}
    gts_or, hyp_or = {}, {}
    g = torch.Generator().manual_seed(0)
    for batch in loader:
        for x in batch:
            x['feats'] = {k: v.to(device) for k, v in x['feats'].items()}
        res = model(batch)
        for r, x in zip(res, batch):
            gts, texts = r['gt_segments'], r['gt_text']
            k = min(len(gts), r['segments'].shape[0])
            preds = r['segments'][:k].tolist()
            emb = r['embeds'][:k].to(device).float()
            emb_pt = r['embeds_pt'][:k].to(device).float()
            n_pred += k
            cap_raw = x['cap_emb'].to(device).float()
            cap_proj = F.normalize(model.embed_head.clip_proj(cap_raw), dim=-1) if same_space else None
            cap_n = F.normalize(cap_raw, dim=-1)
            if full:
                # caption the GT spans themselves: measures captioning alone
                for gi in range(len(gts)):
                    s = pool_f @ r['embeds_gt'][gi].to(device).float()
                    if omni:
                        s = s + o_txt @ r['embeds_gt_omni'][gi].to(device).float()
                    i = mbr_pick(s, pool_n, scale)[1]
                    ret['oracle'].append(float(pool_n[i] @ cap_n[gi]))
                    okey = len(gts_or); gts_or[okey] = [texts[gi]]; hyp_or[okey] = [pool_text[i]]
            for gi, gt in enumerate(gts):
                n_gt += 1
                if not preds:
                    best_ious.append(0.0); continue
                vals = [iou(p, gt) for p in preds]
                j = int(np.argmax(vals)); best = vals[j]
                best_ious.append(best)
                for t in ths:
                    cov[t] += (best >= t)
                if best < 0.3:
                    continue
                q = F.normalize(emb[j], dim=-1)
                if cap_proj is not None:   # GT caption vectors exist only in ONE-PEACE space
                    cos_gt.append(float(q @ cap_proj[gi]))
                ridx = int(torch.randint(len(pool_f), (1,), generator=g))
                cos_rand.append(float(q @ pool_f[ridx]))
                key = len(gts_txt); gts_txt[key] = [texts[gi]]
                s_op = pool_f @ q
                s = s_op
                if omni:
                    s_om = o_txt @ r['embeds_omni'][j].to(device).float()
                    s = s_op + s_om
                t1, mb = mbr_pick(s, pool_n, scale)
                picks = {'top1': t1, 'mbr': mb}
                if full:
                    picks['pt_top1'] = int((pool_f @ F.normalize(emb_pt[j], dim=-1)).argmax())
                    if omni:
                        picks['mbr_onepeace'] = mbr_pick(s_op, pool_n, scale)[1]
                        picks['mbr_omni'] = mbr_pick(s_om, pool_n, scale)[1]
                for v, i in picks.items():
                    ret[v].append(float(pool_n[i] @ cap_n[gi]))
                    hyp[v][key] = [pool_text[i]]
    model.train()
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    cider = lambda v: Cider().compute_score(gts_txt, hyp[v])[0] * 100 if hyp[v] else 0.0
    out = {'R@%.1f' % t: 100.0 * cov[t] / max(n_gt, 1) for t in ths}
    out['mIoU'] = 100.0 * mean(best_ious)
    out['seg/vid'] = n_pred / max(len(loader.dataset), 1)
    out['cos_gt'] = mean(cos_gt) if cos_gt else float('nan')   # not defined outside ONE-PEACE caption space
    out['cos_rand'] = mean(cos_rand)
    out['ret_sim'] = mean(ret['mbr'])
    out['CIDEr'] = cider('mbr')
    if full:
        for v in ('top1', 'pt_top1') + (('mbr_onepeace', 'mbr_omni') if omni else ()):
            out['ret_sim[%s]' % v] = mean(ret[v])
            out['CIDEr[%s]' % v] = cider(v)
        m = meteor(gts_txt, hyp['mbr'])
        if m is not None:
            out['METEOR'] = m
        out['ret_sim[oracle]'] = mean(ret['oracle'])
        out['CIDEr[oracle]'] = Cider().compute_score(gts_or, hyp_or)[0] * 100
        out['n_captioned'] = len(gts_txt)
    return out


# One log row per epoch. Each column: (header, metric key, width, format)
GROUPS = [
    ('train loss', [('event', 'ev_loss', 6, '.3f'), ('reg', 'reg_loss', 6, '.3f'), ('iou', 'iou_loss', 6, '.3f'),
                    ('embed', 'emb_loss', 6, '.3f'), ('span', 'span_loss', 6, '.3f'),
                    ('total', 'final_loss', 6, '.3f')]),
    ('segmentation (val)', [('R@0.3', 'R@0.3', 6, '.2f'), ('R@0.5', 'R@0.5', 6, '.2f'),
                            ('R@0.7', 'R@0.7', 6, '.2f'), ('mIoU', 'mIoU', 6, '.2f'),
                            ('seg/vid', 'seg/vid', 7, '.2f')]),
    ('captioning (val)', [('cos_gt', 'cos_gt', 6, '.3f'), ('cos_rand', 'cos_rand', 8, '.3f'),
                          ('ret_sim', 'ret_sim', 7, '.3f'), ('CIDEr', 'CIDEr', 6, '.2f')]),
]
LEAD = '{:>6} {:>5} {:>8}'


def _cells(fn):
    return ' | '.join(' '.join(fn(c) for c in cols) for _, cols in GROUPS)


HEAD = LEAD.format('epoch', 'time', 'lr') + ' | ' + _cells(lambda c: '%*s' % (c[2], c[0]))
RULE = '-' * (len(HEAD) + 4)


def use_omni_columns():
    """Add the two OmniRetriever teacher losses to the table."""
    global HEAD, RULE
    GROUPS[0][1][5:5] = [('o_txt', 'omni_txt_loss', 6, '.3f'), ('o_av', 'omni_av_loss', 6, '.3f')]
    HEAD = LEAD.format('epoch', 'time', 'lr') + ' | ' + _cells(lambda c: '%*s' % (c[2], c[0]))
    RULE = '-' * (len(HEAD) + 4)


def print_header():
    print('Legend       : S = new best R@0.5, C = new best CIDEr (each saves its checkpoint)')
    print('               cos_rand = cosine to a random caption (floor), ret_sim = similarity of')
    print('               chosen caption to GT caption (ONE-PEACE space), CIDEr = word overlap')
    print(RULE)
    widths = [sum(c[2] for c in cols) + len(cols) - 1 for _, cols in GROUPS]
    print(LEAD.format('', '', '') + ' | ' + ' | '.join(
        '{:^{}}'.format(name, w) for (name, _), w in zip(GROUPS, widths)))
    print(HEAD)
    print(RULE, flush=True)


def print_row(ep, t, lr, vals, mark):
    row = LEAD.format(ep, '%.0fs' % t, '%.1e' % lr) + ' | ' + _cells(
        lambda c: format(vals.get(c[1], float('nan')), '>%d%s' % (c[2], c[3])))
    print(row + ('  ' + mark if mark else ''), flush=True)


def build_loaders(cfg, rng):
    ds = YouCook2CaptionDataset(True, cfg['train_split'], **cfg['dataset'])
    dv = YouCook2CaptionDataset(False, cfg['val_split'], **cfg['dataset'])
    dl = DataLoader(ds, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                    sampler=RandomSampler(ds), collate_fn=trivial_batch_collator,
                    worker_init_fn=worker_init_reset_seed, drop_last=True, generator=rng,
                    persistent_workers=cfg['num_workers'] > 0)
    dlv = DataLoader(dv, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                     sampler=SequentialSampler(dv), collate_fn=trivial_batch_collator)
    return ds, dv, dl, dlv


def eval_only(a, model, dlv, dev, pool_raw, pool_text):
    """Score one checkpoint under several NMS / ranking settings, with all variants."""
    ck = load_weights(model, a.eval)
    print('Checkpoint   : %s (epoch %d)' % (a.eval, ck['epoch'] + 1), flush=True)
    powers = a.iou_power.split(',') if a.iou_power else [str(model.iou_power)]
    for power in powers:
        model.iou_power = float(power)
        for spec in a.nms.split(','):
            kind, *rest = spec.split(':')
            model.nms_cfg = {'soft': kind == 'soft', 'iou': float(rest[0]),
                             'sigma': float(rest[1]) if len(rest) > 1 else 0.5}
            m = evaluate(model, dlv, dev, pool_raw, pool_text, full=not a.fast)
            print('\niou_power %s  NMS %-14s' % (power, spec)
                  + '  '.join('%s %.3f' % (k, v) for k, v in m.items()), flush=True)


def apply_overrides(cfg, items):
    """--set a.b.c=value overrides the config; values are parsed as YAML (numbers, bools, lists)."""
    for it in items:
        key, val = it.split('=', 1)
        node, parts = cfg, key.split('.')
        for k in parts[:-1]:
            node = node.setdefault(k, {})
        node[parts[-1]] = yaml.safe_load(val)
        print('Override     : %s = %r' % (key, node[parts[-1]]), flush=True)


def git_state():
    """Commit hash and whether tracked source files have uncommitted changes.

    Build artefacts (*.egg-info) are ignored: installing nms_1d_cpu rewrites them.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    run = lambda *c: subprocess.run(c, cwd=here, capture_output=True, text=True).stdout.strip()
    try:
        changed = [l for l in run('git', 'status', '--porcelain', '--untracked-files=no').splitlines()
                   if '.egg-info/' not in l]
        return {'commit': run('git', 'rev-parse', '--short', 'HEAD'),
                'branch': run('git', 'rev-parse', '--abbrev-ref', 'HEAD'),
                'dirty': bool(changed)}
    except OSError:
        return {'commit': 'unknown', 'branch': 'unknown', 'dirty': None}


class RunRecord:
    """Everything needed to compare and reproduce a run, saved after every epoch."""

    def __init__(self, a, cfg, out_dir):
        self.path = os.path.join(RUNS_DIR, '%s-%s.json' % (socket.gethostname(), a.output))
        os.makedirs(RUNS_DIR, exist_ok=True)
        self.data = {
            'run': a.output, 'host': socket.gethostname(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
            'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'finished': None,
            'git': git_state(), 'argv': sys.argv, 'overrides': a.set, 'note': a.note,
            'epochs': a.epochs, 'out_dir': out_dir,
            'config': cfg, 'history': [], 'best': {}, 'final_eval': None,
        }
        self.save()

    @classmethod
    def load(cls, run):
        rec = cls.__new__(cls)
        rec.path = os.path.join(RUNS_DIR, '%s-%s.json' % (socket.gethostname(), run))
        with open(rec.path) as f:
            rec.data = json.load(f)
        return rec

    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.data, f, indent=1, default=str)


def main(a):
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    apply_overrides(cfg, a.set)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = fix_random_seed(cfg.get('init_rand_seed', 1234567891), include_cuda=True)
    out_dir = os.path.join(cfg['output_folder'], a.output)

    ds, dv, dl, dlv = build_loaders(cfg, rng)
    print('Data         : %d train / %d val videos' % (len(ds), len(dv)), flush=True)
    # the input width follows the feature source; stored in the run record's config
    cfg['model']['input_dim_V'], cfg['model']['input_dim_A'] = ds.feat_dims
    print('Features     : InternVideo2 %s + BEATs a768, %d row(s)/s, visual %d + audio %d channels' % (
        '+'.join(ds.iv2_video_keys), ds.iv2_rows_per_sec, *ds.feat_dims), flush=True)

    if ds.omni:
        cfg['model']['omni_dim'] = ds.omni_dim
        use_omni_columns()
    cfg['model']['clip_dim'] = ds.pool_train.shape[1]   # caption space (dataset.caption_space)
    model = EventCaptionTransformer(**cfg['model']).to(dev)
    print('Parameters   : %.1fM' % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)

    # the caption pool comes from the TRAIN split only; val captions are never used
    pool_raw = torch.from_numpy(ds.pool_train).to(dev)
    model.set_caption_pool(pool_raw)
    if ds.caption_space != 'onepeace':
        model.pool_metric = torch.from_numpy(ds.pool_emb).to(dev)
        print('Captions     : trained and picked in %s space (%d-d); metrics in ONE-PEACE space'
              % (ds.caption_space, pool_raw.shape[1]), flush=True)
    if ds.omni:
        model.set_omni_pool(torch.from_numpy(ds.omni_text_pool).to(dev),
                            torch.from_numpy(ds.omni_text_ok).to(dev),
                            torch.from_numpy(ds.omni_av_pool).to(dev),
                            torch.as_tensor(ds.av_text_idx, dtype=torch.long, device=dev))
    print('Caption pool : %d unique train captions' % len(ds.pool_text), flush=True)
    if a.eval:
        return eval_only(a, model, dlv, dev, pool_raw, ds.pool_text)
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
    n_ep = a.epochs + cfg['opt']['warmup_epochs']
    best = {'R@0.5': (-1.0, 0, 'best_seg'), 'CIDEr': (-1.0, 0, 'best_cap')}
    print_header()
    for ep in range(n_ep):
        t0 = time.time(); acc = {}
        for batch in dl:
            for x in batch:
                x['feats'] = {k: v.to(dev) for k, v in x['feats'].items()}
            L = model(batch)
            opt.zero_grad(set_to_none=True); L['final_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train_cfg']['clip_grad_l2norm'])
            opt.step(); sch.step()
            for k, v in L.items(): acc[k] = acc.get(k, 0.0) + v.detach().item()
        m = evaluate(model, dlv, dev, pool_raw, ds.pool_text)
        mark = ''
        for key, (val, _, name) in best.items():
            if m[key] > val:
                best[key] = (m[key], ep, name); mark += 'S' if key == 'R@0.5' else 'C'
                torch.save({'epoch': ep, 'state_dict': model.state_dict(), 'metrics': m},
                           os.path.join(out_dir, name + '.pth.tar'))
        vals = dict(m, **{k: v / len(dl) for k, v in acc.items()})
        print_row('%d/%d' % (ep + 1, n_ep), time.time() - t0, sch.get_last_lr()[0], vals, mark)
        rec.data['history'].append(dict(vals, epoch=ep + 1, seconds=round(time.time() - t0, 1)))
        rec.data['best'] = {k: {'value': v, 'epoch': e + 1, 'ckpt': n} for k, (v, e, n) in best.items()}
        rec.save()
    print(RULE)
    for key, (val, ep, name) in best.items():
        print('Best %-6s = %6.2f at epoch %d -> %s' % (
            key, val, ep + 1, os.path.join(out_dir, name + '.pth.tar')), flush=True)

    finalize(rec, model, dlv, dev, pool_raw, ds.pool_text)


def finalize(rec, model, dlv, dev, pool_raw, pool_text):
    """Full evaluation of the run's best captioning checkpoint, stored in the run record."""
    load_weights(model, os.path.join(rec.data['out_dir'], 'best_cap.pth.tar'))
    final = evaluate(model, dlv, dev, pool_raw, pool_text, full=True)
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
    p.add_argument('--nms', default='soft:0.7:0.5',
                   help='NMS settings to score with --eval, comma separated, e.g. soft:0.7:0.5,hard:0.5')
    p.add_argument('--iou-power', default='',
                   help='--eval: IoU-score weights to try, comma separated; empty keeps the config value')
    p.add_argument('--fast', action='store_true', help='--eval: skip the slow variants')
    p.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                   help='override the config, e.g. --set init_rand_seed=2 model.train_cfg.loss_weight_omni_av=0.5')
    p.add_argument('--note', default='', help='free text stored in the run record')
    p.add_argument('--finalize', action='store_true',
                   help='only redo the final evaluation of the finished run --output (same --set as training)')
    main(p.parse_args())
