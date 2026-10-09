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

Distributed download (2026-10-05: Colab and Kaggle can download from YouTube without cookies):
  PC     : python tools/coin_hub.py plan
           splits the COIN videos not downloaded yet into chunks of 100 ids -> dl/plan.json
           (chunk numbers from 1000, so they never collide with the PC's shards 0000..)
  any    : python tools/coin_hub.py fetch --name colab-dl --jobs 4 [--cookies FILE]
           claims a chunk (dl/claims/CCCC__<name>), downloads it with yt-dlp, uploads
           videos/coin_videos_CCCC.tar (the extraction workers take it like any shard) together with
           dl/done/CCCC.json (ok / failed / retry ids), then the next chunk. When YouTube blocks the
           IP ("not a bot", 429) the chunk is released and the machine waits --block-wait minutes.
           The fetcher that finds every chunk done writes videos/DONE (the PC flushes its own partial
           shard first: upload --flush).

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

REPO = os.environ.get('COIN_HUB_REPO', 'nguyenminh04/coin-feats')   # features, claims, plans
VREPO = os.environ.get('COIN_HUB_VREPO', 'nguyenminh04/coin-videos')   # video shards (separate repo for every dataset)
PFX = 'coin'                                                        # shard file prefix: <PFX>_videos_NNNN.tar
SHARD = 100
FORMAT = 'bv*[height<=480][ext=mp4]+ba[ext=m4a]/b[height<=480]/bv*+ba/b'   # as datasets/annotations/download_videos.py
# --dataset anet|htstep: videos in one HF repo, features (+ claims, plan, annotations) in another. 360p keeps shards small.
DATASETS = {
    'anet': dict(feats='nguyenminh04/anet-feats', videos='nguyenminh04/anet-videos', pfx='anet'),
    'htstep': dict(feats='nguyenminh04/htstep-feats', videos='nguyenminh04/htstep-videos', pfx='htstep'),
}
FORMAT360 = 'bv*[height<=360][ext=mp4]+ba[ext=m4a]/b[height<=360]/bv*+ba/b'


def configure(a):
    global REPO, VREPO, PFX, FORMAT
    if a.dataset != 'coin':
        d = DATASETS[a.dataset]
        REPO, VREPO, PFX, FORMAT = d['feats'], d['videos'], d['pfx'], FORMAT360
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
    vfiles = files if VREPO == REPO else set(api.list_repo_files(VREPO, repo_type='dataset'))
    shards = sorted(int(m.group(1)) for f in vfiles for m in [re.match(r'videos/%s_videos_(\d+)\.tar$' % PFX, f)] if m)
    done = {int(m.group(1)) for f in files for m in [re.match(r'feats/%s_feats_(\d+)\.tar$' % PFX, f)] if m}
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
    for r in (REPO, VREPO):
        api.create_repo(r, repo_type='dataset', private=True, exist_ok=True)
    tmp = a.tmp or os.path.join(ROOT, 'logs', 'coin_tmp')
    os.makedirs(tmp, exist_ok=True)
    while True:
        active = not (a.final or a.flush)
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
                            repo_id=VREPO, repo_type='dataset', commit_message='videos shard %04d' % k)
            log('uploaded shard %04d: %d videos, %.0f MB, %.1f MB/s' % (k, len(part), mb, mb / (time.time() - t0)))
            os.remove(tar)
        api.upload_file(path_or_fileobj=json.dumps(manifest).encode(), path_in_repo='videos/manifest.json',
                        repo_id=VREPO, repo_type='dataset', commit_message='manifest')
        if a.flush:
            log('flushed: %d videos in %d shards, no DONE (the fetchers write it)' % (len(ids), n_shards))
            return
        if not active:
            api.upload_file(path_or_fileobj=json.dumps({'videos': len(ids), 'shards': n_shards}).encode(),
                            path_in_repo='videos/DONE', repo_id=REPO, repo_type='dataset', commit_message='DONE')
            log('download finished: %d videos in %d shards, DONE written' % (len(ids), n_shards))
            return
        log('%d videos downloaded, %d full shards uploaded; next check in %d min' % (len(ids), n_shards, a.every))
        time.sleep(a.every * 60)


def plan(a):
    """dl/plan.json: COIN ids neither downloaded by the PC nor failed for good, in chunks of SHARD.
    --retry: dl/plan_retry.json instead, from the chunks already fetched: their 'retry' ids and the
    failures that look transient (RETRY), in chunks from --first (use 2000); run it with fetch --plan
    dl/plan_retry.json on the PC, with cookies."""
    api = hub()
    if a.retry:
        from huggingface_hub import hf_hub_download
        done, _, _ = dl_state(api)
        todo = []
        for k in sorted(done):
            r = json.load(open(hf_hub_download(REPO, 'dl/done/%04d.json' % k, repo_type='dataset', token=api.token)))
            todo += list(r['retry']) + [v for v, why in r['failed'].items() if any(m in why for m in RETRY)]
        chunks = {'%04d' % (a.first + i): todo[i * SHARD:(i + 1) * SHARD] for i in range(-(-len(todo) // SHARD))}
        api.upload_file(path_or_fileobj=json.dumps(chunks).encode(), path_in_repo='dl/plan_retry.json', repo_id=REPO,
                        repo_type='dataset', commit_message='retry plan: %d videos' % len(todo))
        log('dl/plan_retry.json: %d videos from %d fetched chunks, %d chunks' % (len(todo), len(done), len(chunks)))
        return
    if PFX != 'coin':
        return plan_other(api, a)
    db = json.load(open(os.path.join(COIN, 'COIN.json')))['database']
    skip = set()
    for f in ('downloaded.txt', 'failed.txt'):
        if os.path.exists(os.path.join(COIN, f)):
            skip |= {l.split('\t')[0].strip() for l in open(os.path.join(COIN, f)) if l.strip()}
    todo = [v for v in db if v not in skip]
    chunks = {'%04d' % (a.first + i): todo[i * SHARD:(i + 1) * SHARD] for i in range(-(-len(todo) // SHARD))}
    api.upload_file(path_or_fileobj=json.dumps(chunks).encode(), path_in_repo='dl/plan.json', repo_id=REPO,
                    repo_type='dataset', commit_message='download plan: %d videos in %d chunks' % (len(todo), len(chunks)))
    log('dl/plan.json: %d videos in %d chunks (%s..%s)' % (len(todo), len(chunks), min(chunks), max(chunks)))


def plan_other(api, a):
    """anet: --n random train ids (+ --n-val from val_1) of ActivityNet Captions; htstep: all val_seen (the only labelled val split) + --n train.
    Writes dl/plan.json and anno/<dataset>.json (the annotations of the planned videos only) to the features repo."""
    import random
    for r in (REPO, VREPO):
        api.create_repo(r, repo_type='dataset', private=not a.public, exist_ok=True)
    if PFX == 'anet':
        d = os.path.join(ROOT, 'datasets', 'Anet')
        tr, va = json.load(open(os.path.join(d, 'train.json'))), json.load(open(os.path.join(d, 'val_1.json')))
        ok = lambda v: len(v['sentences']) == len(v['timestamps']) > 0 and all(0 <= t[0] < t[1] for t in v['timestamps'])
        rng = random.Random(2026)
        pick = lambda db, n, role: [(k[2:], role, db[k]) for k in rng.sample(sorted(k for k in db if ok(db[k])), n)]
        if a.extend:   # keep the plan and annotations on the hub, add --n new train videos as chunks after the last one
            from huggingface_hub import hf_hub_download
            get = lambda f: json.load(open(hf_hub_download(REPO, f, repo_type='dataset', token=api.token)))
            old, oldplan = get('anno/anet.json'), get('dl/plan.json')
            rng = random.Random(len(old))
            sel = pick({k: v for k, v in tr.items() if k[2:] not in old}, a.n, 'train')
        else:
            sel = pick(tr, a.n, 'train') + pick(va, a.n_val, 'val')
        anno = {i: dict(role=r, duration=v['duration'], segments=v['timestamps'], captions=v['sentences'])
                for i, r, v in sel}
    else:
        d = os.path.join(ROOT, 'datasets', 'htstep', 'data')
        ann, spl = json.load(open(os.path.join(d, 'annotations.json'))), json.load(open(os.path.join(d, 'video_splits.json')))
        import csv
        head = {int(r['global_step_index']): r['headline'] for r in csv.DictReader(open(os.path.join(d, 'taxonomy.csv')))}
        anno = {}
        rng = random.Random(2026)
        for role, i in [('val_seen', i) for i in spl['val_seen']] + [('train', i) for i in rng.sample(sorted(spl['train']), a.n)]:
            v = ann.get(i)
            segs = [x for x in (v or {}).get('annotations', []) if x['partial'] == 'Full Match' and x['id'] in head]
            if segs:
                anno[i] = dict(role=role, duration=v['duration'], activity=v['activity'],
                               segments=[x['segment'] for x in segs], captions=[head[x['id']] for x in segs])
    ids = list(anno)
    first = a.first
    chunks = {}
    if PFX == 'anet' and a.extend:
        first, chunks, anno = int(max(oldplan)) + 1, dict(oldplan), dict(old, **anno)
    chunks.update({'%04d' % (first + i): ids[i * SHARD:(i + 1) * SHARD] for i in range(-(-len(ids) // SHARD))})
    ids = list(anno)
    for path, obj in (('dl/plan.json', chunks), ('anno/%s.json' % PFX, anno)):
        api.upload_file(path_or_fileobj=json.dumps(obj).encode(), path_in_repo=path, repo_id=REPO,
                        repo_type='dataset', commit_message='%s: %d videos' % (path, len(ids)))
    log('%s: %d videos in %d chunks -> %s (videos: %s)' % (PFX, len(ids), len(chunks), REPO, VREPO))


def dl_state(api):
    """Chunks finished (dl/done/CCCC.json) and download claims {chunk: [(worker, age_h)], oldest first}."""
    files = set(api.list_repo_files(REPO, repo_type='dataset'))
    done = {int(m.group(1)) for f in files for m in [re.match(r'dl/done/(\d+)\.json$', f)] if m}
    claims = {}
    if any(f.startswith('dl/claims/') for f in files):
        now = datetime.now(timezone.utc)
        for e in api.list_repo_tree(REPO, path_in_repo='dl/claims', repo_type='dataset', expand=True):
            m = re.match(r'dl/claims/(\d+)__(.+)$', e.path)
            if m and e.last_commit is not None:
                claims.setdefault(int(m.group(1)), []).append((m.group(2), (now - e.last_commit.date).total_seconds() / 3600))
    for v in claims.values():
        v.sort(key=lambda c: -c[1])
    return done, claims, files


BLOCK = ('not a bot', 'HTTP Error 429', 'Too Many Requests', 'rate-limit', 'rate limit')
# not permanent: try again later (age gate / login: from the PC with cookies; 403 and empty files are transient)
RETRY = ('sign in', 'Sign in', 'Interrupted', 'cookies are no longer valid', 'HTTP Error 403', 'downloaded file is empty')


def fetch_one(vid, out_dir, a):
    """Download one video -> ('ok' | 'block' | 'retry' | 'failed', reason)."""
    import shutil
    ytdlp = shutil.which('yt-dlp') or os.path.join(os.path.dirname(sys.executable), 'yt-dlp')
    out = os.path.join(out_dir, vid + '.mp4')
    cmd = [ytdlp, '--quiet', '--no-warnings', '--no-progress', '-f', FORMAT, '--merge-output-format', 'mp4',
           '-o', out, '--sleep-interval', str(a.sleep), '--max-sleep-interval', str(a.sleep * 2)]
    if a.ffmpeg:
        cmd += ['--ffmpeg-location', a.ffmpeg]
    if a.cookies:
        cmd += ['--cookies', a.cookies]
    env = dict(os.environ, PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ['PATH'])   # deno
    r = subprocess.run(cmd + ['https://www.youtube.com/watch?v=' + vid], capture_output=True, text=True, env=env)
    reason = (r.stderr.strip().splitlines() or ['?'])[-1][:200]
    if r.returncode == 0 and os.path.exists(out):
        return 'ok', ''
    for f in os.listdir(out_dir):   # leftovers of a failed merge
        if f.startswith(vid + '.') and f != vid + '.mp4':
            os.remove(os.path.join(out_dir, f))
    if any(b in r.stderr for b in BLOCK):
        return 'block', reason
    if any(m in r.stderr for m in RETRY):
        return 'retry', reason
    return 'failed', reason


def fetch(a):
    import shutil
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, hf_hub_download
    api = hub()
    tmp = a.tmp or '/tmp/coin_fetch_%s' % a.name
    chunks = json.load(open(hf_hub_download(REPO, a.plan, repo_type='dataset', token=api.token,
                                            force_download=True)))
    while True:
        done, claims, files = dl_state(api)
        todo = sorted(int(k) for k in chunks if int(k) not in done)
        if not todo:
            if 'videos/DONE' not in files and a.write_done:
                api.upload_file(path_or_fileobj=json.dumps({'chunks': len(chunks)}).encode(),
                                path_in_repo='videos/DONE', repo_id=REPO, repo_type='dataset',
                                commit_message='DONE (all download chunks finished)')
                log('every chunk downloaded: videos/DONE written')
            log('ALL_FETCHED: %d chunks' % len(chunks)); return
        free = [k for k in todo if owner(claims, k, a.stale) in (None, a.name)]
        if not free:
            log('no free chunk (%d left, all claimed); waiting' % len(todo)); time.sleep(300); continue
        k = free[0] if not a.reverse else free[-1]
        claim = 'dl/claims/%04d__%s' % (k, a.name)
        api.upload_file(path_or_fileobj=('%s %d' % (a.name, time.time())).encode(), path_in_repo=claim, repo_id=REPO, repo_type='dataset',
                        commit_message='dl claim %04d %s' % (k, a.name))
        time.sleep(5)
        done, claims, _ = dl_state(api)
        if k in done or owner(claims, k, a.stale) != a.name:
            log('chunk %04d taken by %s, next' % (k, owner(claims, k, a.stale))); continue
        ids = chunks['%04d' % k]
        out_dir = os.path.join(tmp, 'c%04d' % k)
        os.makedirs(out_dir, exist_ok=True)
        t0 = time.time()
        res = {}
        with ThreadPoolExecutor(a.jobs) as ex:
            blocked = []
            def job(v):
                if blocked:
                    return
                if os.path.exists(os.path.join(out_dir, v + '.mp4')):
                    res[v] = ('ok', ''); return
                st = fetch_one(v, out_dir, a)
                if st[0] == 'block':
                    blocked.append(v)
                res[v] = st
            list(ex.map(job, ids))
        if blocked:
            n_ok = sum(1 for s in res.values() if s[0] == 'ok')
            log('chunk %04d: BLOCKED by YouTube after %d ok (%s); releasing it, waiting %d min'
                % (k, n_ok, res[blocked[0]][1][:120], a.block_wait))
            api.delete_file(claim, repo_id=REPO, repo_type='dataset', commit_message='dl release %04d' % k)
            time.sleep(a.block_wait * 60); continue   # downloaded files stay in out_dir for the next try
        ok = [v for v in ids if res.get(v, ('?',))[0] == 'ok']
        tar = os.path.join(tmp, '%s_videos_%04d.tar' % (PFX, k))
        with tarfile.open(tar, 'w') as t:
            for v in ok:
                t.add(os.path.join(out_dir, v + '.mp4'), arcname=v + '.mp4')
        report = {'by': a.name, 'ok': ok,
                  'failed': {v: r for v, (s, r) in res.items() if s == 'failed'},
                  'retry': {v: r for v, (s, r) in res.items() if s == 'retry'}}
        if ok:   # videos first: dl/done is the proof that the shard is there
            api.create_commit(VREPO, [CommitOperationAdd('videos/%s_videos_%04d.tar' % (PFX, k), tar)],
                              repo_type='dataset', commit_message='videos chunk %04d (%s)' % (k, a.name))
        api.create_commit(REPO, [CommitOperationAdd('dl/done/%04d.json' % k, json.dumps(report, indent=0).encode())],
                          repo_type='dataset', commit_message='dl done %04d (%s)' % (k, a.name))
        mb = os.path.getsize(tar) / 1e6
        log('chunk %04d done by %s: %d ok, %d failed, %d retry, %.0f MB, %.1f min' % (
            k, a.name, len(ok), len(report['failed']), len(report['retry']), mb, (time.time() - t0) / 60))
        shutil.rmtree(out_dir, ignore_errors=True); os.remove(tar)


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
        api.upload_file(path_or_fileobj=('%s %d' % (a.name, time.time())).encode(), path_in_repo='claims/%04d__%s' % (k, a.name),
                        repo_id=REPO, repo_type='dataset', commit_message='claim %04d %s' % (k, a.name))
        time.sleep(5)   # two workers claiming the same shard at once: the older claim wins
        _, done, claims, _ = state(api)
        if k in done or owner(claims, k, a.stale) != a.name:
            log('shard %04d taken by %s, next' % (k, owner(claims, k, a.stale))); continue
        log('shard %04d: downloading' % k)
        src = hf_hub_download(VREPO, 'videos/%s_videos_%04d.tar' % (PFX, k), repo_type='dataset', local_dir=tmp,
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
        tar = os.path.join(tmp, '%s_feats_%04d.tar' % (PFX, k))
        with tarfile.open(tar, 'w') as t:
            for f in npz:
                t.add(os.path.join(out, f), arcname=f)
            info = tarfile.TarInfo('failed.txt'); data = '\n'.join(failed).encode(); info.size = len(data)
            t.addfile(info, io.BytesIO(data))
        api.upload_file(path_or_fileobj=tar, path_in_repo='feats/%s_feats_%04d.tar' % (PFX, k), repo_id=REPO,
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
    p.add_argument('cmd', choices=('upload', 'work', 'status', 'plan', 'fetch'))
    p.add_argument('--dataset', default='coin', choices=('coin', 'anet', 'htstep'))
    p.add_argument('--n', type=int, default=3000, help='plan: train videos (anet 3000, htstep use 900)'); p.add_argument('--n-val', type=int, default=500)
    p.add_argument('--public', action='store_true', help='plan: create the repos public')
    p.add_argument('--name', default='worker'); p.add_argument('--tmp', default='')
    p.add_argument('--every', type=int, default=10, help='upload: minutes between checks')
    p.add_argument('--final', action='store_true', help='upload: download finished, write the last shard and DONE')
    p.add_argument('--stale', type=float, default=6.0, help='hours after which a claim is dead')
    p.add_argument('--repo', default=''); p.add_argument('--video-ckpt', default='')
    p.add_argument('--audio-ckpt', default=''); p.add_argument('--workers', type=int, default=4)
    p.add_argument('--flush', action='store_true', help='upload: also the last partial shard, but no DONE')
    p.add_argument('--first', type=int, default=1000, help='plan: number of the first chunk')
    p.add_argument('--retry', action='store_true', help='plan: dl/plan_retry.json from the fetched chunks')
    p.add_argument('--extend', action='store_true', help='plan (anet): add --n new train videos after the existing chunks')
    p.add_argument('--plan', default='dl/plan.json', help='fetch: the plan to work through')
    p.add_argument('--jobs', type=int, default=3, help='fetch: parallel yt-dlp processes')
    p.add_argument('--sleep', type=int, default=2, help='fetch: seconds yt-dlp waits before a download (x2 max)')
    p.add_argument('--cookies', default='', help='fetch: cookies.txt for yt-dlp (PC only)')
    p.add_argument('--ffmpeg', default='', help='fetch: ffmpeg binary when not on PATH')
    p.add_argument('--block-wait', type=int, default=30, help='fetch: minutes to wait after a YouTube block')
    p.add_argument('--reverse', action='store_true', help='fetch: take chunks from the end (fewer claim races)')
    p.add_argument('--no-done', dest='write_done', action='store_false', help='fetch: never write videos/DONE')
    a = p.parse_args()
    configure(a)
    if a.cmd == 'fetch' and a.stale == 6.0:
        a.stale = 2.0   # a download chunk takes minutes to an hour
    {'upload': upload, 'work': work, 'status': status, 'plan': plan, 'fetch': fetch}[a.cmd](a)
