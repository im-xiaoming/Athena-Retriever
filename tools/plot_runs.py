"""Plot training curves and final metrics: original model vs improved (no teacher) vs teacher.

  python tools/plot_runs.py

Reads per-epoch metrics from experiments/runs/*.json, or from logs/<run>.log for runs
whose record has no history (backfilled runs and the original-code baseline).
Writes experiments/plots/{curves,final,losses}.png.

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
GROUPS = [  # (label, colour, runs)
    ('Original model', '#2a78d6', ['orig_baseline']),
    ('Improved, no teacher', '#eb6834', ['sched10', 'base_seed2']),
    ('Improved + OmniRetriever teacher', '#1baf7a',
     ['omni65', 'omni100', 'omni100_seed2', 'omni100_w05', 'omni100_avonly']),
]
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
        xs = [row['epoch'] for row in rows if key in row]
        ys = [row[key] for row in rows if key in row]
        if xs:
            out.append((r, np.array(xs), np.array(ys)))
    return out


def plot_curves(hist, panels, fname, title):
    n = len(panels)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow), facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()
    for ax, (key, label) in zip(axes, panels):
        style(ax, label)
        for gname, color, runs in GROUPS:
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
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=10,
               labelcolor=INK, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha='left', y=1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, fname), dpi=130, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def plot_final(hist, panels, fname, title):
    """Epoch-12 value of every run as a dot, group mean as a short line; no checkpoint picking.

    Dots and mean ticks instead of bars: the y axis is zoomed to the data, and bars on
    a truncated axis would exaggerate the gaps.
    """
    short = {'Original model': 'Original', 'Improved, no teacher': 'No teacher',
             'Improved + OmniRetriever teacher': 'Teacher'}
    fig, axes = plt.subplots(1, len(panels), figsize=(2.9 * len(panels), 3.6), facecolor=SURFACE)
    for ax, (key, label) in zip(axes, panels):
        style(ax, label)
        ax.grid(axis='x', visible=False)
        allv, empty = [], []
        for gi, (gname, color, runs) in enumerate(GROUPS):
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
        ax.set_xlim(-0.5, len(GROUPS) - 0.3)
        for gi in empty:   # placed in axes coordinates so it never stretches the figure
            ax.text(gi, 0.5, 'n/a', transform=ax.get_xaxis_transform(), ha='center', color=INK2, fontsize=8)
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels([short[g[0]] for g in GROUPS], fontsize=8, color=INK2)
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=130, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    hist = load_histories()
    missing = [r for _, _, runs in GROUPS for r in runs if r not in hist]
    if missing:
        print('no history for:', ', '.join(missing))
    plot_curves(hist, [('R@0.5', 'Recall @ IoU 0.5 (val, %)'), ('R@0.7', 'Recall @ IoU 0.7 (val, %)'),
                       ('mIoU', 'mean best IoU (val, %)'), ('ret_sim', 'ret_sim: chosen vs GT caption (val)'),
                       ('cos_gt', 'cos_gt: segment vs GT caption (val)'), ('CIDEr', 'CIDEr (val)')],
                'curves.png', 'Validation metrics per epoch')
    plot_curves(hist, [('ev_loss', 'event loss (train)'), ('reg_loss', 'boundary loss (train)'),
                       ('final_loss', 'total loss (train, not comparable across groups)')],
                'losses.png', 'Training losses per epoch')
    plot_final(hist, [('R@0.5', 'R@0.5'), ('R@0.7', 'R@0.7'), ('mIoU', 'mIoU'),
                      ('ret_sim', 'ret_sim'), ('cos_gt', 'cos_gt'), ('CIDEr', 'CIDEr')],
               'final.png', 'Last epoch (12): every run as a dot, line = group mean')
    print('wrote', ', '.join(sorted(os.listdir(OUT))))


if __name__ == '__main__':
    main()
