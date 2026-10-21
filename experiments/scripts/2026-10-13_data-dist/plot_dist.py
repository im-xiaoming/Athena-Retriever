"""How different are the four training sets? Annotation statistics and feature-space overlap.

  ~/uniav-api-env/bin/python experiments/scripts/2026-10-13_data-dist/plot_dist.py

Plot text is in Vietnamese (the user's language). Only videos with InternVideo2 features are counted (train and val pooled). Writes
experiments/plots/data_dist.png (annotations) and experiments/plots/data_feats.png (features), and prints the
numbers behind them.

The model resamples every video to max_seq_len = 256 grid steps (force_upsampling), so it sees a step's length
relative to the video, not in seconds: "grid length" = segment / duration * 256. A point is trained on FPN level l
when its larger distance to the step's boundaries is in regression_range[l]; for the centre point that distance is
half the step length, so the levels split grid length at 8 / 16 / 32 / 64 / 128.
"""
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, ScalarFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, 'experiments', 'plots')
GRID = 256
LEVEL_EDGES = [8, 16, 32, 64, 128]   # grid lengths where the centre point moves to the next FPN level
COLORS = {'YouCook2': '#d62728', 'COIN': '#2ca02c', 'ActivityNet': '#1f77b4', 'HT-Step': '#9467bd'}
FEAT_DIRS = {'YouCook2': 'data/youcookii/iv2_feats', 'COIN': 'data/coin/iv2_feats',
             'ActivityNet': 'data/anet/iv2_feats', 'HT-Step': 'data/htstep/iv2_feats'}


def has_feats(name):
    d = os.path.join(ROOT, FEAT_DIRS[name])
    return {f[:-4] for f in os.listdir(d) if f.endswith('.npz')}


def load():
    """{dataset: [(video id, duration, [[s, e], ...], [text, ...]), ...]}"""
    out = {}
    db = json.load(open(os.path.join(ROOT, 'data/youcookii/annotations/youcookii_annotations_trainval.json')))['database']
    ok = has_feats('YouCook2')
    out['YouCook2'] = [(k, v['duration'], [a['segment'] for a in v['annotations']], [a['sentence'] for a in v['annotations']])
                       for k, v in db.items() if k in ok]
    coin = json.load(open(os.path.join(ROOT, 'data/coin/coin_anno.json')))['videos']
    ok = has_feats('COIN')
    out['COIN'] = [(k, v['duration'], v['segments'], v['labels']) for k, v in coin.items() if k in ok]
    for name, f in (('ActivityNet', 'data/anet/annotations.json'), ('HT-Step', 'data/htstep/annotations.json')):
        db = json.load(open(os.path.join(ROOT, f)))['database']
        ok = has_feats(name)
        out[name] = [(k, v['duration'], [a['segment'] for a in v['annotations']], [a['sentence'] for a in v['annotations']])
                     for k, v in db.items() if k in ok and v['annotations']]
    return out


def tiou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def stats(videos):
    s = {k: [] for k in ('dur', 'seg_s', 'rel', 'grid', 'nseg', 'cover', 'center', 'words')}
    n_ovl = n_seg = 0
    for _, dur, segs, texts in videos:
        segs = [(max(0.0, a), min(dur, b)) for a, b in segs if b > a]
        if not segs or dur <= 0:
            continue
        s['dur'].append(dur)
        s['nseg'].append(len(segs))
        mask = np.zeros(int(np.ceil(dur * 10)) + 1, bool)   # 0.1 s bins for the covered fraction
        for i, (a, b) in enumerate(segs):
            s['seg_s'].append(b - a)
            s['rel'].append((b - a) / dur)
            s['grid'].append((b - a) / dur * GRID)
            s['center'].append((a + b) / 2 / dur)
            mask[int(a * 10):int(b * 10)] = True
            n_seg += 1
            n_ovl += any(tiou((a, b), o) >= 0.3 for j, o in enumerate(segs) if j != i)
        s['cover'].append(mask[:-1].mean())
        s['words'] += [len(re.findall(r"\w+", t)) for t in texts]
    s = {k: np.asarray(v, float) for k, v in s.items()}
    s['ovl'] = n_ovl / max(n_seg, 1)
    s['short8'] = float((s['grid'] < 8).mean())   # steps on the finest FPN level only
    return s


def comma_axes(fig):
    """Decimal comma on the linear axes (Vietnamese number format)."""
    f = FuncFormatter(lambda x, _: ('%g' % x).replace('.', ','))
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):   # numeric linear axes only (not log, not the bar chart's names)
            if type(axis.get_major_formatter()) is ScalarFormatter:
                axis.set_major_formatter(f)


def kde_hist(ax, data, name, bins, log=False):
    # share of the items per bin, not density: with log-spaced bins a density inflates the narrow left bins
    ax.hist(data, bins=bins, weights=np.full(len(data), 1 / len(data)), histtype='step', lw=1.8, color=COLORS[name],
            label='%s (trung vị %s)' % (name, fmt(np.median(data))))
    if log:
        ax.set_xscale('log')


def fmt(x):
    """Vietnamese number format for the plot text: decimal comma."""
    s = '%.0f' % x if x >= 10 else ('%.1f' % x if x >= 1 else '%.2f' % x)
    return s.replace('.', ',')


def plot_annotations(S):
    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    ax = axes.ravel()
    kw = dict()
    lb = np.logspace(np.log10(5), np.log10(3000), 40)
    for n, s in S.items():
        kde_hist(ax[0], s['dur'], n, lb, log=True)
    ax[0].set(title='Độ dài video', xlabel='giây (thang log)')
    lb = np.logspace(np.log10(0.5), np.log10(1000), 40)
    for n, s in S.items():
        kde_hist(ax[1], s['seg_s'], n, lb, log=True)
    ax[1].set(title='Độ dài bước / sự kiện', xlabel='giây (thang log)')
    lb = np.logspace(np.log10(0.002), 0, 40)
    for n, s in S.items():
        kde_hist(ax[2], s['rel'] * 100, n, lb * 100, log=True)
    ax[2].set(title='Độ dài bước so với độ dài video', xlabel='% độ dài video (thang log)')

    lb = np.logspace(np.log10(0.5), np.log10(GRID), 40)
    for n, s in S.items():
        ax[3].hist(s['grid'], bins=lb, weights=np.full(len(s['grid']), 1 / len(s['grid'])), histtype='step', lw=1.8, color=COLORS[n],
                   label='%s (%s%% bước < 8 ô)' % (n, fmt(100 * s['short8'])))
    ax[3].set_xscale('log')
    for i, e in enumerate(LEVEL_EDGES):
        ax[3].axvline(e, color='gray', ls=':', lw=1)
    for i, (lo, hi) in enumerate(zip([0.5] + LEVEL_EDGES, LEVEL_EDGES + [GRID])):
        ax[3].text(np.sqrt(lo * hi), ax[3].get_ylim()[1] * 0.97, 'L%d' % i, ha='center', va='top', color='gray', fontsize=9)
    ax[3].set(title='Thứ model thấy: độ dài bước trên lưới 256 ô\n(nét chấm: ranh giới mức FPN của điểm giữa bước)',
              xlabel='số ô lưới (thang log)')

    mx = max(int(np.percentile(s['nseg'], 99)) for s in S.values())
    b = np.arange(0.5, mx + 1.5, 1)
    for n, s in S.items():
        ax[4].hist(np.minimum(s['nseg'], mx), bins=b, weights=np.full(len(s['nseg']), 1 / len(s['nseg'])), histtype='step', lw=1.8, color=COLORS[n],
                   label='%s (trung vị %d)' % (n, np.median(s['nseg'])))
    ax[4].set(title='Số bước mỗi video', xlabel='số bước')

    b = np.linspace(0, 1, 26)
    for n, s in S.items():
        kde_hist(ax[5], s['cover'], n, b)
    ax[5].set(title='Tỉ lệ thời lượng video được gán nhãn', xlabel='tỉ lệ')

    for n, s in S.items():
        kde_hist(ax[6], s['center'], n, b)
    ax[6].set(title='Vị trí tâm bước trong video', xlabel='vị trí tương đối (0 = đầu, 1 = cuối)')

    names = list(S)
    ax[7].bar(names, [100 * S[n]['ovl'] for n in names], color=[COLORS[n] for n in names])
    for i, n in enumerate(names):
        ax[7].text(i, 100 * S[n]['ovl'] + 0.3, '%s%%' % ('%.1f' % (100 * S[n]['ovl'])).replace('.', ','), ha='center')
    ax[7].set(title='Bước chồng lấn với bước khác (tIoU ≥ 0,3)', ylabel='% số bước', ylim=(0, 115 * max(S[n]['ovl'] for n in names)))

    b = np.arange(0.5, 41.5, 1)
    for n, s in S.items():
        kde_hist(ax[8], np.minimum(s['words'], 40), n, b)
    ax[8].set(title='Số từ mỗi caption / nhãn bước', xlabel='số từ (40 = từ 40 trở lên)')

    for a in ax:
        if a is ax[3]:   # below the L0..L5 level names
            a.legend(fontsize=9, loc='upper left', bbox_to_anchor=(0, 0.92))
        elif a is not ax[7]:
            a.legend(fontsize=9)
            a.set_ylabel('tỉ lệ trong mỗi cột')
        a.grid(alpha=0.25)
    fig.suptitle('Phân phối annotation của bốn bộ dữ liệu huấn luyện (các video có đặc trưng, gộp train + val)', fontsize=14)
    comma_axes(fig)
    fig.tight_layout()
    p = os.path.join(OUT, 'data_dist.png')
    fig.savefig(p, dpi=110)
    print('wrote', p)


def video_means(name, ids, n_max, rng):
    ids = list(ids)
    rng.shuffle(ids)
    v, a = [], []
    for k in ids[:n_max]:
        d = np.load(os.path.join(ROOT, FEAT_DIRS[name], k + '.npz'))
        v.append(d['v768'].astype(np.float32).mean(0))
        a.append(d['a768'].astype(np.float32).mean(0))
    return np.stack(v), np.stack(a)


def pca2(x):
    x = x - x.mean(0)
    _, sv, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T, (sv[:2] ** 2) / (sv ** 2).sum()


def centroid_acc(x, y, rng):
    """Nearest-centroid dataset classification, half train / half test: how separable the datasets are."""
    idx = rng.permutation(len(y))
    tr, te = idx[: len(y) // 2], idx[len(y) // 2:]
    cs = np.stack([x[tr][y[tr] == c].mean(0) for c in range(y.max() + 1)])
    pred = np.argmax(x[te] @ cs.T, 1)
    return float((pred == y[te]).mean())


def plot_features(data, n_max=500):
    rng = np.random.default_rng(0)
    names = list(data)
    V, A, Y = [], [], []
    for i, n in enumerate(names):
        v, a = video_means(n, [k for k, *_ in data[n]], n_max, rng)
        V.append(v), A.append(a), Y.append(np.full(len(v), i))
    Y = np.concatenate(Y)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    for ax, X, key, title in ((axes[0], np.concatenate(V), 'video', 'Hình (InternVideo2 v768)'),
                              (axes[1], np.concatenate(A), 'audio', 'Âm thanh (BEATs a768)')):
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        Xc = X - X.mean(0)
        Xc = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
        acc = centroid_acc(Xc, Y, rng)
        z, ev = pca2(X)
        for i, n in enumerate(names):
            m = Y == i
            ax.scatter(z[m, 0], z[m, 1], s=7, alpha=0.45, color=COLORS[n], label='%s (%d video)' % (n, m.sum()))
        ax.set(title='%s, trung bình theo từng video\nPCA 2 chiều (%.0f%% + %.0f%% phương sai)\nđoán đúng bộ dữ liệu theo tâm cụm: %.0f%% (ngẫu nhiên: 25%%)'
               % (title, 100 * ev[0], 100 * ev[1], 100 * acc), xlabel='thành phần chính 1', ylabel='thành phần chính 2')
        ax.legend(markerscale=3, fontsize=9)
        ax.grid(alpha=0.25)
        print('%s: nearest-centroid dataset accuracy %.3f' % (key, acc))
    comma_axes(fig)
    fig.tight_layout()
    p = os.path.join(OUT, 'data_feats.png')
    fig.savefig(p, dpi=110)
    print('wrote', p)


def main():
    data = load()
    S = {n: stats(v) for n, v in data.items()}
    print('%-12s %6s %8s %8s %8s %8s %8s %8s %8s %8s' % ('dataset', 'videos', 'dur_med', 'seg_med', 'rel_med%',
                                                       'grid_med', '<8cells', 'nseg_med', 'cover', 'overlap'))
    for n, s in S.items():
        print('%-12s %6d %8.0f %8.1f %8.1f %8.1f %7.0f%% %8.0f %8.2f %7.1f%%' % (
            n, len(s['dur']), np.median(s['dur']), np.median(s['seg_s']), 100 * np.median(s['rel']),
            np.median(s['grid']), 100 * s['short8'], np.median(s['nseg']), np.median(s['cover']), 100 * s['ovl']))
    plot_annotations(S)
    plot_features(data)


if __name__ == '__main__':
    main()
