"""How close is the student (our model) to the OmniRetriever-7B teacher?

  python tools/compare_teacher.py

Uses the 500 validation clips that have teacher audio+video vectors in
data/youcookii/omni_emb_full.npz. Every model captions the GT spans themselves, so
only the segment vector is compared, not segmentation. Metrics per model:
  ret_sim, top1 : similarity of the consensus / top-1 caption to the GT caption (ONE-PEACE space)
  CIDEr         : word-overlap metric of the consensus caption
  fidelity      : students trained with the teacher only - cosine between the student's
                  vector projected into the 3584-d teacher space and the teacher's own
                  vector for the same clip (1.0 = the student reproduces the teacher)
  fidelity_floor: the same cosine against the teacher vector of a random other clip
The teacher row picks captions with its clip vector against its caption vectors.
Writes experiments/teacher_vs_student.json and experiments/plots/teacher_vs_student.png.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from train_event import build_loaders, mbr_pick, load_ckpt  # noqa: E402
from libs.modeling import make_multimodal_meta_arch          # noqa: E402

OMNI = os.path.join(ROOT, 'data', 'youcookii', 'omni_emb_full.npz')
STUDENTS = [('sched10', False), ('base_seed2', False),
            ('omni65', True), ('omni100', True), ('omni100_seed2', True)]


def cider(gts, hyp):
    from pycocoevalcap.cider.cider import Cider
    return Cider().compute_score(gts, hyp)[0] * 100


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    om = np.load(OMNI)
    with open(os.path.join(ROOT, 'data', 'youcookii', 'annotations', 'youcookii_annotations_trainval.json')) as f:
        subset = {v: x['subset'] for v, x in json.load(f)['database'].items()}
    # the file also holds train clips; only validation clips are comparable
    val_keys = {k[:-4] for k in om.files if k.endswith('__av') and subset[k[:-4].rsplit('#', 1)[0]] == 'validation'}
    base = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'youcook2_event.yaml')))
    out = {}

    for run, uses_teacher in STUDENTS:
        cfg = json.loads(json.dumps(base))
        if uses_teacher:
            cfg['dataset']['omni_emb_file'] = OMNI
        ds, dv, _, dlv = build_loaders(cfg, torch.Generator().manual_seed(0))
        if uses_teacher:
            cfg['model']['omni_dim'] = ds.omni_dim
        model = make_multimodal_meta_arch('EventCaptionTransformer', **cfg['model']).to(dev)
        model.load_state_dict(load_ckpt(os.path.join(ROOT, 'ckpt', run, 'best_cap.pth.tar'))['state_dict'])
        pool_raw = torch.from_numpy(ds.pool_emb).to(dev)
        model.set_caption_pool(pool_raw)
        if uses_teacher:
            model.set_omni_pool(torch.from_numpy(ds.omni_text_pool).to(dev), torch.from_numpy(ds.omni_text_ok).to(dev),
                                torch.from_numpy(ds.omni_av_pool).to(dev))
        model.eval()
        pool_f = F.normalize(model.embed_head.clip_proj(pool_raw), dim=-1)
        pool_n = F.normalize(pool_raw, dim=-1)
        scale = float(model.embed_head.logit_scale.exp().clamp(max=100))
        sims, top1, fid, fid_rand, gts, hyp = [], [], [], [], {}, {}
        others = sorted(val_keys)
        rng = np.random.default_rng(0)
        with torch.no_grad():
            for batch in dlv:
                for x in batch:
                    x['feats'] = {k: v.to(dev) for k, v in x['feats'].items()}
                for r, x in zip(model(batch), batch):
                    cap_n = F.normalize(x['cap_emb'].to(dev).float(), dim=-1)
                    for gi in range(len(r['gt_segments'])):
                        key = '%s#%d' % (x['video_id'], gi)
                        if key not in val_keys:
                            continue
                        s = pool_f @ r['embeds_gt'][gi].to(dev).float()
                        if uses_teacher:
                            so = r['embeds_gt_omni'][gi].to(dev).float()
                            s = s + model.omni_text.masked_fill(~model.omni_text_ok[:, None], 0) @ so
                            t = F.normalize(torch.from_numpy(om[key + '__av'].astype(np.float32)).to(dev), dim=-1)
                            fid.append(float(so @ t))
                            o = others[rng.integers(len(others))]   # floor: teacher vector of another clip
                            fid_rand.append(float(so @ F.normalize(torch.from_numpy(om[o + '__av'].astype(np.float32)).to(dev), dim=-1)))
                        t1, mb = mbr_pick(s, pool_n, scale)
                        sims.append(float(pool_n[mb] @ cap_n[gi])); top1.append(float(pool_n[t1] @ cap_n[gi]))
                        gts[len(gts)] = [r['gt_text'][gi]]; hyp[len(hyp)] = [ds.pool_text[mb]]
        out[run] = {'teacher_trained': uses_teacher, 'n': len(sims), 'ret_sim': float(np.mean(sims)),
                    'top1': float(np.mean(top1)), 'CIDEr': cider(gts, hyp),
                    'fidelity': float(np.mean(fid)) if fid else None,
                    'fidelity_floor': float(np.mean(fid_rand)) if fid_rand else None}
        print(run, out[run], flush=True)
        del model

    # teacher alone: its clip vector against its caption vectors, same consensus pick
    cap = np.load(os.path.join(ROOT, 'data', 'youcookii', 'caption_emb.npz'), allow_pickle=True)
    text_of = dict(zip([str(k) for k in cap['keys']], [str(t) for t in cap['sentences']]))
    emb_of = dict(zip([str(k) for k in cap['keys']], cap['emb']))
    vec_by_text = {text_of[f[:-6]]: om[f] for f in om.files if f.endswith('__text')}
    pool_text = ds.pool_text
    OT = F.normalize(torch.tensor(np.stack([vec_by_text[t] for t in pool_text])).float(), dim=-1).to(dev)
    pool_n = F.normalize(torch.from_numpy(ds.pool_emb).to(dev), dim=-1)
    sims, top1, gts, hyp = [], [], {}, {}
    for key in sorted(val_keys):
        q = F.normalize(torch.from_numpy(om[key + '__av'].astype(np.float32)).to(dev), dim=-1)
        t1, mb = mbr_pick(OT @ q, pool_n, 50.0)
        gt = F.normalize(torch.from_numpy(emb_of[key].astype(np.float32)).to(dev), dim=-1)
        sims.append(float(pool_n[mb] @ gt)); top1.append(float(pool_n[t1] @ gt))
        gts[len(gts)] = [text_of[key]]; hyp[len(hyp)] = [pool_text[mb]]
    out['teacher'] = {'teacher_trained': None, 'n': len(sims), 'ret_sim': float(np.mean(sims)),
                      'top1': float(np.mean(top1)), 'CIDEr': cider(gts, hyp), 'fidelity': 1.0, 'fidelity_floor': None}
    print('teacher', out['teacher'], flush=True)
    with open(os.path.join(ROOT, 'experiments', 'teacher_vs_student.json'), 'w') as f:
        json.dump(out, f, indent=1)
    plot(out)


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
    groups = [('No teacher', '#eb6834', [k for k, v in out.items() if v['teacher_trained'] is False]),
              ('Student (with teacher)', '#1baf7a', [k for k, v in out.items() if v['teacher_trained']])]
    teacher_c = '#2a78d6'
    panels = [('ret_sim', 'ret_sim (consensus pick)'), ('top1', 'ret_sim (top-1 pick)'),
              ('CIDEr', 'CIDEr'), ('fidelity', 'fidelity: cos(student, teacher)')]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.9), facecolor=SURFACE)
    for ax, (key, label) in zip(axes, panels):
        ax.set_facecolor(SURFACE); ax.set_title(label, color=INK, fontsize=11, loc='left')
        ax.grid(True, axis='y', color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'): ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        vals_all = []
        for gi, (gname, color, runs) in enumerate(groups):
            vals = [out[r][key] for r in runs if out[r][key] is not None]
            if not vals:
                ax.text(gi, 0.5, 'n/a', transform=ax.get_xaxis_transform(), ha='center', color=INK2, fontsize=8)
                continue
            vals_all += vals
            ax.scatter(np.full(len(vals), gi) + np.linspace(-0.1, 0.1, len(vals)), vals, s=42, color=color,
                       edgecolors=SURFACE, linewidths=1.5, zorder=3)
            m = float(np.mean(vals))
            ax.hlines(m, gi - 0.28, gi + 0.28, color=color, linewidth=2.2)
            ax.annotate('%.3g' % m, (gi + 0.3, m), va='center', fontsize=8, color=INK)
        if key == 'fidelity':
            fl = float(np.mean([v['fidelity_floor'] for v in out.values() if v.get('fidelity_floor') is not None]))
            ax.axhline(fl, color=INK2, linewidth=1.2, linestyle=(0, (1, 2)))
            ax.annotate('floor (other clip) %.2f' % fl, (1.45, fl), va='bottom', ha='right', fontsize=8, color=INK2)
            vals_all.append(fl)
        tv = out['teacher'][key]
        ax.axhline(tv, color=teacher_c, linewidth=1.6, linestyle=(0, (4, 3)))
        ax.annotate('teacher %.3g' % tv, (1.45, tv), va='bottom', ha='right', fontsize=8, color=INK)
        lo, hi = min(vals_all + [tv]), max(vals_all + [tv])
        pad = (hi - lo) * 0.3 + 1e-6
        ax.set_ylim(lo - pad, hi + pad); ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 1]); ax.set_xticklabels([g[0] for g in groups], fontsize=8, color=INK2)
    fig.suptitle('Student vs OmniRetriever-7B teacher on the same 500 GT val clips (dashed = teacher)',
                 color=INK, fontsize=12, x=0.01, ha='left')
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, 'experiments', 'plots'), exist_ok=True)
    fig.savefig(os.path.join(ROOT, 'experiments', 'plots', 'teacher_vs_student.png'), dpi=130,
                bbox_inches='tight', facecolor=SURFACE)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--plot-only':
        plot(json.load(open(os.path.join(ROOT, 'experiments', 'teacher_vs_student.json'))))
    else:
        main()
