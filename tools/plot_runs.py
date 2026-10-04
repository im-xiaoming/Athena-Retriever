"""Plot training curves and final metrics of the runs, grouped.

  python tools/plot_runs.py

Reads per-epoch metrics from experiments/runs/*.json, or from logs/<run>.log for runs
whose record has no history (backfilled runs and the original-code baseline).
Writes experiments/plots/:
  {curves,final,losses}.png            original model -> improved -> teacher -> InternVideo2
  variants_{curves,final,losses}.png   variants of the InternVideo2 model
  round3_{curves,final,losses}.png     512 steps, OmniRetriever text as caption space, T+V+A teacher target
  features_{curves,final,losses}.png   input features under the same teacher setup, full data:
                                       ONE-PEACE vs InternVideo2 (and both concatenated)

Groups: each run is a thin line in its group's colour, the bold line is the group mean.
The original model's ret_sim is its own top-1 pick (it had no consensus pick) and it
reports no CIDEr; that is what that system actually outputs.
"""
import glob
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'experiments', 'plots')

# reference palette, categorical slots 1-3 (validated all-pairs on the light surface)
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
# colour follows the entity in every figure: aqua = ONE-PEACE + teacher, yellow = InternVideo2 + teacher
GROUPS = [  # (label, colour, runs)
    ('Original model', '#2a78d6', ['orig_baseline']),
    ('Improved, no teacher', '#eb6834', ['sched10', 'base_seed2']),
    ('Improved + OmniRetriever teacher', '#1baf7a',
     ['omni65', 'omni100', 'omni100_seed2', 'omni100_w05', 'omni100_avonly']),
    ('InternVideo2 + teacher', '#eda100', ['iv2', 'iv2_seed2']),
]
# the same teacher (full coverage) and data, only the input features differ
FEATURE_GROUPS = [
    ('ONE-PEACE', '#1baf7a', ['omni100', 'omni100_seed2']),
    ('InternVideo2 v768', '#eda100', ['iv2', 'iv2_seed2']),
    ('InternVideo2 v768 + v512', '#e87ba4', ['iv2_v512']),
    ('InternVideo2 + ONE-PEACE', '#4a3aa7', ['iv2op']),
]
# variants of the InternVideo2 model (all with the teacher unless noted)
VARIANT_GROUPS = [
    ('iv2 (default)', '#eda100', ['iv2', 'iv2_seed2']),
    ('no teacher', '#2a78d6', ['iv2_noteach']),
    ('dropout 0.1, drop-path 0.2', '#eb6834', ['iv2_reg']),
    ('emb / span loss 0.4', '#e87ba4', ['iv2_emb04']),
    ('8 epochs', '#008300', ['iv2_ep8', 'iv2_ep8_seed2']),
    ('8 epochs + dropout', '#4a3aa7', ['iv2_ep8_reg']),
]
# round 3: resolution, caption space and teacher target (single runs, iv2 = 2 seeds)
ROUND3_GROUPS = [
    ('iv2 (default)', '#eda100', ['iv2', 'iv2_seed2']),
    ('512 steps', '#2a78d6', ['iv2_len512']),
    ('teacher text as caption space', '#eb6834', ['txt_omni']),
    ('T+V+A teacher target', '#e87ba4', ['teach_tva']),
]
SHORT = {'Original model': 'Original', 'Improved, no teacher': 'No teacher',
         'Improved + OmniRetriever teacher': 'Teacher', 'InternVideo2 + teacher': 'InternVideo2',
         'ONE-PEACE': 'ONE-PEACE', 'InternVideo2 v768': 'IV2', 'InternVideo2 v768 + v512': 'IV2 +v512',
         'InternVideo2 + ONE-PEACE': 'IV2 +OP', 'iv2 (default)': 'iv2', 'no teacher': 'no teacher',
         'dropout 0.1, drop-path 0.2': 'dropout', 'emb / span loss 0.4': 'emb 0.4', '8 epochs': '8 ep',
         '8 epochs + dropout': '8 ep +drop', '512 steps': '512 steps',
         'teacher text as caption space': 'omni text', 'T+V+A teacher target': 'T+V+A'}
HEAD_TO_KEY = {'event': 'ev_loss', 'reg': 'reg_loss', 'iou': 'iou_loss', 'embed': 'emb_loss',
               'span': 'span_loss', 'o_txt': 'omni_txt_loss', 'o_av': 'omni_av_loss', 'total': 'final_loss'}


def parse_log(path):
    """Epoch rows of the English table log -> list of dicts."""
    rows, cols = [], None
    for line in open(path, errors='ignore'):
        if line.lstrip().startswith('epoch') and 'time' in line:
            cols = [HEAD_TO_KEY.get(t, t) for t in line.replace('|', ' ').split()]
        elif cols and re.match(r'\s*\d+/\d+\s', line):
            toks = line.replace('|', ' ').split()
            r = {}
            for c, t in zip(cols, toks):
                if c in ('time', 'lr'):
                    continue
                if c == 'epoch':
                    r['epoch'] = int(t.split('/')[0])
                else:
                    try:
                        r[c] = float(t)
                    except ValueError:
                        pass
            rows.append(r)
    return rows


def load_histories():
    hist = {}
    for f in glob.glob(os.path.join(ROOT, 'experiments', 'runs', '*.json')):
        d = json.load(open(f))
        if d.get('history'):
            hist[d['run']] = d['history']
    for f in glob.glob(os.path.join(ROOT, 'logs', '*.log')):
        run = os.path.basename(f)[:-4]
        if run not in hist and not run.endswith('.final'):
            rows = parse_log(f)
            if rows:
                hist[run] = rows
    return hist


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc='left', pad=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def series(hist, runs, key):
    out = []
    for r in runs:
        rows = hist.get(r) or []
        # cos_gt does not exist when captions live outside ONE-PEACE space (older records stored 0)
        ok = [row for row in rows if key in row and not (key == 'cos_gt' and not row[key])
              and row[key] == row[key]]
        xs = [row['epoch'] for row in ok]
        ys = [row[key] for row in ok]
        if xs:
            out.append((r, np.array(xs), np.array(ys)))
    return out


def plot_curves(hist, panels, fname, title, groups=GROUPS):
    n = len(panels)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow), facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()
    for ax, (key, label) in zip(axes, panels):
        style(ax, label)
        for gname, color, runs in groups:
            ss = series(hist, runs, key)
            if not ss:
                continue
            for _, x, y in ss:
                ax.plot(x, y, color=color, linewidth=1.0, alpha=0.45)
            L = min(len(x) for _, x, _ in ss)
            mx, my = ss[0][1][:L], np.mean([y[:L] for _, _, y in ss], axis=0)
            ax.plot(mx, my, color=color, linewidth=2.2, label='%s (%d run%s)' % (gname, len(ss), 's' if len(ss) > 1 else ''))
            ax.plot(mx[-1:], my[-1:], 'o', color=color, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_xlabel('epoch', color=INK2, fontsize=9)
    for ax in axes[n:]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=min(len(groups), 4), frameon=False, fontsize=10,
               labelcolor=INK, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha='left', y=1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, fname), dpi=130, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def plot_final(hist, panels, fname, title, groups=GROUPS):
    """Epoch-12 value of every run as a dot, group mean as a short line; no checkpoint picking.

    Dots and mean ticks instead of bars: the y axis is zoomed to the data, and bars on
    a truncated axis would exaggerate the gaps.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6), facecolor=SURFACE)
    for ax, (key, label) in zip(axes, panels):
        style(ax, label)
        ax.grid(axis='x', visible=False)
        allv, empty = [], []
        for gi, (gname, color, runs) in enumerate(groups):
            vals = [y[-1] for _, _, y in series(hist, runs, key)]
            if not vals:
                empty.append(gi); continue
            allv += vals
            m = float(np.mean(vals))
            ax.scatter(np.full(len(vals), gi) + np.linspace(-0.1, 0.1, len(vals)), vals,
                       s=40, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=3)
            ax.hlines(m, gi - 0.28, gi + 0.28, color=color, linewidth=2.2, zorder=2)
            ax.annotate('%.3g' % m, (gi + 0.3, m), va='center', fontsize=8, color=INK)
        lo, hi = min(allv), max(allv)
        pad = (hi - lo) * 0.35 + 1e-6
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(-0.5, len(groups) - 0.3)
        for gi in empty:   # placed in axes coordinates so it never stretches the figure
            ax.text(gi, 0.5, 'n/a', transform=ax.get_xaxis_transform(), ha='center', color=INK2, fontsize=8)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([SHORT[g[0]] for g in groups], fontsize=7.5, color=INK2, rotation=20)
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=130, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    hist = load_histories()
    missing = [r for groups in (GROUPS, FEATURE_GROUPS) for _, _, runs in groups for r in runs if r not in hist]
    if missing:
        print('no history for:', ', '.join(sorted(set(missing))))
    curves = [('R@0.5', 'Recall @ IoU 0.5 (val, %)'), ('R@0.7', 'Recall @ IoU 0.7 (val, %)'),
              ('mIoU', 'mean best IoU (val, %)'), ('ret_sim', 'ret_sim: chosen vs GT caption (val)'),
              ('cos_gt', 'cos_gt: segment vs GT caption (val)'), ('CIDEr', 'CIDEr (val)')]
    finals = [('R@0.5', 'R@0.5'), ('R@0.7', 'R@0.7'), ('mIoU', 'mIoU'),
              ('ret_sim', 'ret_sim'), ('cos_gt', 'cos_gt'), ('CIDEr', 'CIDEr')]
    for prefix, groups, what in (('', GROUPS, ''),
                                 ('features_', FEATURE_GROUPS, ' - input features, same teacher, full data'),
                                 ('variants_', VARIANT_GROUPS, ' - variants of the InternVideo2 model'),
                                 ('round3_', ROUND3_GROUPS, ' - resolution, caption space, teacher target')):
        plot_curves(hist, curves, prefix + 'curves.png', 'Validation metrics per epoch' + what, groups)
        plot_curves(hist, [('ev_loss', 'event loss (train)'), ('reg_loss', 'boundary loss (train)'),
                           ('final_loss', 'total loss (train, not comparable across groups)')],
                    prefix + 'losses.png', 'Training losses per epoch' + what, groups)
        plot_final(hist, finals, prefix + 'final.png',
                   'Last epoch: every run as a dot, line = group mean' + what, groups)
    print('wrote', ', '.join(sorted(os.listdir(OUT))))


if __name__ == '__main__':
    main()
