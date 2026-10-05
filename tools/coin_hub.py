"""COIN feature extraction shared by several GPU workers through one private HF dataset repo.

The PC downloads COIN from YouTube (datasets/annotations/download_videos.py); cloud GPUs cannot
(YouTube blocks their IPs). This script moves the work through HF:

  PC     : python tools/coin_hub.py upload
           packs finished downloads into shards of 100 videos -> videos/coin_videos_NNNN.tar,
           every 10 min, until stopped. When the download is complete, run it once with --final:
           it also uploads the last partial shard and writes videos/DONE.
  worker : HF_TOKEN=... python tools/coin_hub.py work --name colab-t4 \\
               --repo /content/InternVideo --video-ckpt .../InternVideo2-stage2_1b-224p-f4.pt \\
               --audio-ckpt .../audio_6b.pth
           claims a shard nobody is working on (claims/NNNN__<name>), extracts it with
           tools/extract_internvideo2.py, uploads feats/coin_feats_NNNN.tar (one .npz per video +
           failed.txt), then takes the next one; exits when videos/DONE exists and every shard is done.
  status : python tools/coin_hub.py status

Any number of workers can run (Kaggle 2x T4: two processes with CUDA_VISIBLE_DEVICES=0 / 1 and
different --name). A claim older than --stale hours is considered dead and can be taken over.
"""
import argparse
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone

REPO = os.environ.get('COIN_HUB_REPO', 'nguyenminh04/coin-data')   # override only for tests
SHARD = 100
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COIN = os.path.join(ROOT, 'datasets', 'annotations')


def hub():
    from huggingface_hub import HfApi
    token = os.environ.get('HF_TOKEN')
    if not token and os.path.exists(os.path.join(ROOT, '.env')):
        for line in open(os.path.join(ROOT, '.env')):
            if line.startswith('HF_TOKEN='):
                token = line.strip().split('=', 1)[1]
    assert token, 'HF_TOKEN not set'
    return HfApi(token=token)


def log(*a):
    print(time.strftime('%m-%d %H:%M:%S'), *a, flush=True)


def state(api):
    """Shards uploaded, shards finished, claims {shard: [(worker, age_h), ...] oldest first}, DONE flag."""
    files = set(api.list_repo_files(REPO, repo_type='dataset'))
    shards = sorted(int(m.group(1)) for f in files for m in [re.match(r'videos/coin_videos_(\d+)\.tar$', f)] if m)
    done = {int(m.group(1)) for f in files for m in [re.match(r'feats/coin_feats_(\d+)\.tar$', f)] if m}
    claims = {}
    if any(f.startswith('claims/') for f in files):
        now = datetime.now(timezone.utc)
        for e in api.list_repo_tree(REPO, path_in_repo='claims', repo_type='dataset', expand=True):
            m = re.match(r'claims/(\d+)__(.+)$', e.path)
            if m and e.last_commit is not None:
                age = (now - e.last_commit.date).total_seconds() / 3600
                claims.setdefault(int(m.group(1)), []).append((m.group(2), age))
    for v in claims.values():
        v.sort(key=lambda c: -c[1])
    return shards, done, claims, 'videos/DONE' in files


def owner(claims, k, stale):
    """The worker holding shard k: the oldest claim that is not stale, or None."""
    live = [w for w, age in claims.get(k, []) if age <= stale]
    return live[0] if live else None


# ---------------------------------------------------------------------------------------- PC side
def upload(a):
    api = hub()
    api.create_repo(REPO, repo_type='dataset', private=True, exist_ok=True)
    tmp = a.tmp or os.path.join(ROOT, 'logs', 'coin_tmp')
    os.makedirs(tmp, exist_ok=True)
    while True:
        active = not a.final
        ids = list(dict.fromkeys(l.strip() for l in open(os.path.join(COIN, 'downloaded.txt')) if l.strip()))
        paths = {}
        for d, _, fs in os.walk(os.path.join(COIN, 'videos')):
            for f in fs:
                if f.endswith('.mp4'):
                    paths[f[:-4]] = os.path.join(d, f)
        ids = [i for i in ids if i in paths]
        n_shards = len(ids) // SHARD if active else -(-len(ids) // SHARD)
        have, _, _, _ = state(api)
        manifest = {}
        for k in range(n_shards):
            part = ids[k * SHARD:(k + 1) * SHARD]
            manifest['%04d' % k] = part
            if k in have:
                continue
            tar = os.path.join(tmp, 'coin_videos_%04d.tar' % k)
            with tarfile.open(tar, 'w') as t:
                for i in part:
                    t.add(paths[i], arcname=i + '.mp4')
            t0, mb = time.time(), os.path.getsize(tar) / 1e6
            api.upload_file(path_or_fileobj=tar, path_in_repo='videos/coin_videos_%04d.tar' % k,
                            repo_id=REPO, repo_type='dataset', commit_message='videos shard %04d' % k)
            log('uploaded shard %04d: %d videos, %.0f MB, %.1f MB/s' % (k, len(part), mb, mb / (time.time() - t0)))
            os.remove(tar)
        api.upload_file(path_or_fileobj=json.dumps(manifest).encode(), path_in_repo='videos/manifest.json',
                        repo_id=REPO, repo_type='dataset', commit_message='manifest')
        if not active:
            api.upload_file(path_or_fileobj=json.dumps({'videos': len(ids), 'shards': n_shards}).encode(),
                            path_in_repo='videos/DONE', repo_id=REPO, repo_type='dataset', commit_message='DONE')
            log('download finished: %d videos in %d shards, DONE written' % (len(ids), n_shards))
            return
        log('%d videos downloaded, %d full shards uploaded; next check in %d min' % (len(ids), n_shards, a.every))
        time.sleep(a.every * 60)


# ------------------------------------------------------------------------------------ worker side
def work(a):
    from huggingface_hub import hf_hub_download
    api = hub()
    tmp = a.tmp or '/tmp/coin_work_%s' % a.name
    os.makedirs(tmp, exist_ok=True)
    while True:
        shards, done, claims, finished = state(api)
        free = [k for k in shards if k not in done and owner(claims, k, a.stale) in (None, a.name)]
        if not free:
            if finished and set(shards) <= done:
                log('ALL_DONE: %d shards' % len(shards)); return
            log('nothing free (%d uploaded, %d done, %d claimed); waiting' % (len(shards), len(done), len(claims)))
            time.sleep(300); continue
        k = free[0]
        api.upload_file(path_or_fileobj=a.name.encode(), path_in_repo='claims/%04d__%s' % (k, a.name),
                        repo_id=REPO, repo_type='dataset', commit_message='claim %04d %s' % (k, a.name))
        time.sleep(5)   # two workers claiming the same shard at once: the older claim wins
        _, done, claims, _ = state(api)
        if k in done or owner(claims, k, a.stale) != a.name:
            log('shard %04d taken by %s, next' % (k, owner(claims, k, a.stale))); continue
        log('shard %04d: downloading' % k)
        src = hf_hub_download(REPO, 'videos/coin_videos_%04d.tar' % k, repo_type='dataset', local_dir=tmp,
                              token=api.token)
        vids, out = os.path.join(tmp, 'v%04d' % k), os.path.join(tmp, 'f%04d' % k)
        with tarfile.open(src) as t:
            t.extractall(vids)
        os.remove(src)
        cmd = [sys.executable, '-u', os.path.join(ROOT, 'tools', 'extract_internvideo2.py'), '--videos', vids,
               '--out', out, '--repo', a.repo, '--video-ckpt', a.video_ckpt, '--audio-ckpt', a.audio_ckpt,
               '--workers', str(a.workers)]
        failed, t0 = [], time.time()
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as p:
            for line in p.stdout:
                print(line, end='', flush=True)
                if line.startswith(('FAIL ', 'SKIP ')):
                    failed.append(line.strip())
        if p.returncode != 0:
            log('shard %04d: extractor exited with %d; retrying later' % (k, p.returncode)); time.sleep(60); continue
        npz = sorted(f for f in os.listdir(out) if f.endswith('.npz'))
        tar = os.path.join(tmp, 'coin_feats_%04d.tar' % k)
        with tarfile.open(tar, 'w') as t:
            for f in npz:
                t.add(os.path.join(out, f), arcname=f)
            info = tarfile.TarInfo('failed.txt'); data = '\n'.join(failed).encode(); info.size = len(data)
            t.addfile(info, io.BytesIO(data))
        api.upload_file(path_or_fileobj=tar, path_in_repo='feats/coin_feats_%04d.tar' % k, repo_id=REPO,
                        repo_type='dataset', commit_message='feats %04d (%s)' % (k, a.name))
        log('shard %04d done by %s: %d videos, %d failed, %.0f min' % (k, a.name, len(npz), len(failed),
                                                                       (time.time() - t0) / 60))
        subprocess.run(['rm', '-rf', vids, out, tar])


def status(a):
    shards, done, claims, finished = state(hub())
    log('%d shards uploaded%s, %d done' % (len(shards), ' (all: DONE)' if finished else '', len(done)))
    for k in sorted(claims):
        if k not in done:
            w, age = claims[k][0]
            log('  shard %04d: %s, claimed %.1f h ago%s' % (k, w, age, ' (stale)' if age > a.stale else ''))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=('upload', 'work', 'status'))
    p.add_argument('--name', default='worker'); p.add_argument('--tmp', default='')
    p.add_argument('--every', type=int, default=10, help='upload: minutes between checks')
    p.add_argument('--final', action='store_true', help='upload: download finished, write the last shard and DONE')
    p.add_argument('--stale', type=float, default=6.0, help='hours after which a claim is dead')
    p.add_argument('--repo', default=''); p.add_argument('--video-ckpt', default='')
    p.add_argument('--audio-ckpt', default=''); p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    {'upload': upload, 'work': work, 'status': status}[a.cmd](a)
