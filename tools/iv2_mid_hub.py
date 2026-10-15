"""Several machines extract InternVideo2 middle-layer features for YouCook2 together, coordinated on HF.

  python tools/iv2_mid_hub.py work --name colab-a100 --repo /content/InternVideo \\
      --video-ckpt /content/iv2_ckpt/InternVideo2-stage2_1b-224p-f4.pt
  python tools/iv2_mid_hub.py status

The 1500 videos of HF nguyenminh04/youcook2-videos (sorted) are cut into chunks of --chunk videos.
A worker claims a free chunk (iv2_mid_work/claims/CCCC__<name> in the features repo), downloads its
videos, runs extract_internvideo2.mid_features on each and uploads iv2_mid/<id>.npz plus
iv2_mid_work/done/CCCC.json in one commit, then releases the claim. Two workers claiming the same chunk:
the older claim wins. A claim older than --stale hours is dead and can be taken over. Needs HF_TOKEN
(gated InternVideo2 weights, private features repo).
"""
import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np

VIDEOS = 'nguyenminh04/youcook2-videos'
FEATS = 'nguyenminh04/uniav-youcook2-data'
WORK = 'iv2_mid_work'


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def video_list(api):
    return sorted(f for f in api.list_repo_files(VIDEOS, repo_type='dataset') if f.startswith('videos/') and f.endswith('.mp4'))


def state(api):
    """Chunks done, and claims {chunk: [(worker, age_h)], oldest first}."""
    done, claims = set(), {}
    now = datetime.now(timezone.utc)
    for sub in ('done', 'claims'):
        try:
            entries = list(api.list_repo_tree(FEATS, path_in_repo='%s/%s' % (WORK, sub), repo_type='dataset', expand=True))
        except Exception:   # folder not created yet
            continue
        for e in entries:
            name = e.path.rsplit('/', 1)[1]
            if sub == 'done':
                done.add(int(name.split('.')[0]))
            elif re.match(r'\d+__.+$', name):
                k, w = name.split('__', 1)
                claims.setdefault(int(k), []).append((w, (now - e.last_commit.date).total_seconds() / 3600))
    for v in claims.values():
        v.sort(key=lambda c: -c[1])
    return done, claims


def owner(claims, k, stale):
    live = [w for w, age in claims.get(k, []) if age <= stale]
    return live[0] if live else None


def fetch(api, files, d):
    """Download one chunk's videos into d; returns the local paths."""
    from huggingface_hub import hf_hub_download
    os.makedirs(d, exist_ok=True)
    return [hf_hub_download(VIDEOS, f, repo_type='dataset', local_dir=d) for f in files]


def work(a):
    from huggingface_hub import HfApi, CommitOperationAdd
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    import extract_internvideo2 as ex
    api = HfApi()
    files = video_list(api)
    chunks = [files[i:i + a.chunk] for i in range(0, len(files), a.chunk)]
    log('%d videos, %d chunks of %d' % (len(files), len(chunks), a.chunk))
    dev = torch.device('cuda:%d' % a.gpu)
    torch.cuda.set_device(dev)
    enc, _ = ex.build_video_model(a.repo, a.video_ckpt, dev)
    pool = ThreadPoolExecutor(a.workers)
    t0, secs = time.time(), 0
    while True:
        done, claims = state(api)
        free = [k for k in range(len(chunks)) if k not in done and owner(claims, k, a.stale) in (None, a.name)]
        if not free:
            if len(done) == len(chunks):
                log('ALL_DONE'); return
            log('nothing free (%d/%d done); waiting' % (len(done), len(chunks))); time.sleep(300); continue
        k = {'asc': free[0], 'desc': free[-1]}.get(a.order) or random.choice(free)
        claim = '%s/claims/%04d__%s' % (WORK, k, a.name)
        api.upload_file(path_or_fileobj=b'%d' % time.time(), path_in_repo=claim, repo_id=FEATS, repo_type='dataset',
                        commit_message='iv2_mid claim %04d %s' % (k, a.name))
        time.sleep(5)
        done, claims = state(api)
        if k in done or owner(claims, k, a.stale) != a.name:
            log('chunk %04d taken by %s, next' % (k, owner(claims, k, a.stale))); continue
        d = os.path.join(a.tmp, '%04d' % k)
        log('chunk %04d: downloading %d videos' % (k, len(chunks[k])))
        paths = fetch(api, chunks[k], d)
        out = os.path.join(d, 'out')
        os.makedirs(out, exist_ok=True)
        # bounded prefetch: decoded videos are ~200 MB each
        futures = {i: pool.submit(ex.ffmpeg_frames, paths[i], ffmpeg=a.ffmpeg) for i in range(min(a.workers, len(paths)))}
        ok, failed = [], []
        for i, p in enumerate(paths):
            vid = os.path.basename(p)[:-4]
            fu = futures.pop(i)
            if i + a.workers < len(paths):
                futures[i + a.workers] = pool.submit(ex.ffmpeg_frames, paths[i + a.workers], ffmpeg=a.ffmpeg)
            try:   # one bad video must not stop the chunk
                frames = fu.result()
                if len(frames) == 0:
                    raise RuntimeError('no frames')
                mid = ex.mid_features(enc, frames, dev, a.layers)
            except Exception as e:
                log('FAIL %s: %r' % (vid, e)); failed.append(vid); continue
            np.savez(os.path.join(out, vid + '.npz'), **{n: v.astype(np.float16) for n, v in mid.items()})
            ok.append(vid)
            secs += len(frames) / 2
        ops = [CommitOperationAdd('iv2_mid/%s.npz' % v, os.path.join(out, v + '.npz')) for v in ok]
        ops.append(CommitOperationAdd('%s/done/%04d.json' % (WORK, k), json.dumps(
            {'worker': a.name, 'layers': a.layers, 'ok': ok, 'failed': failed}).encode()))
        for attempt in range(5):
            try:
                api.create_commit(FEATS, ops, repo_type='dataset', commit_message='iv2_mid chunk %04d (%s)' % (k, a.name))
                break
            except Exception as e:
                log('upload failed (%r), retry' % e); time.sleep(60 * (attempt + 1))
        else:
            raise RuntimeError('chunk %04d not uploaded' % k)
        try:
            api.delete_file(claim, repo_id=FEATS, repo_type='dataset', commit_message='iv2_mid release %04d' % k)
        except Exception as e:
            log('release failed: %r' % e)
        shutil.rmtree(d, ignore_errors=True)
        el = time.time() - t0
        log('chunk %04d done: %d ok, %d failed | %.1f video-s/s | %d/%d chunks done' % (
            k, len(ok), len(failed), secs / el, len(done) + 1, len(chunks)))


def status(a):
    from huggingface_hub import HfApi
    api = HfApi()
    n = -(-len(video_list(api)) // a.chunk)
    done, claims = state(api)
    log('%d/%d chunks done' % (len(done), n))
    for k in sorted(claims):
        for w, age in claims[k]:
            log('  chunk %04d: %s, claimed %.1f h ago%s' % (k, w, age, ' (stale)' if age > a.stale else ''))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=('work', 'status'))
    p.add_argument('--name', default='pc')
    p.add_argument('--repo', default='InternVideo', help='upstream InternVideo clone')
    p.add_argument('--video-ckpt', default='ckpt/internvideo2/InternVideo2-stage2_1b-224p-f4.pt')
    p.add_argument('--layers', type=int, nargs='+', default=[20, 30, 35])
    p.add_argument('--chunk', type=int, default=20)
    p.add_argument('--order', choices=('asc', 'desc', 'random'), default='random')
    p.add_argument('--stale', type=float, default=3.0)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--workers', type=int, default=4, help='decode threads')
    p.add_argument('--tmp', default='/tmp/iv2_mid')
    p.add_argument('--ffmpeg', default='ffmpeg')
    a = p.parse_args()
    {'work': work, 'status': status}[a.cmd](a)
