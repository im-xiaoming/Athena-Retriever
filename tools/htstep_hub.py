"""HT-Step download and feature extraction shared by several machines through two private HF dataset repos.

Download (Colab and Kaggle can download from YouTube without cookies; the PC needs them):
  PC     : python tools/htstep_hub.py plan [--n 900]
           all val_seen videos (the only labelled val split) + --n random train videos of HT-Step
           (datasets/htstep/data) in chunks of 100 ids -> dl/plan.json, and their annotations -> anno/htstep.json
           (chunk numbers from --first, 1000 by default)
  any    : python tools/htstep_hub.py fetch --name colab-dl --jobs 4 [--cookies FILE]
           claims a chunk (dl/claims/CCCC__<name>), downloads it with yt-dlp, uploads
           videos/htstep_videos_CCCC.tar (the extraction workers take it like any shard) together with
           dl/done/CCCC.json (ok / failed / retry ids), then the next chunk. When YouTube blocks the
           IP ("not a bot", 429) the chunk is released and the machine waits --block-wait minutes.
           The fetcher that finds every chunk done writes videos/DONE (unless --no-done).
  PC     : python tools/htstep_hub.py plan --retry --first 2000
           dl/plan_retry.json: the 'retry' ids and transient failures of the fetched chunks; run it with
           fetch --plan dl/plan_retry.json, with cookies.
Extraction (--model picks the features; each model has its own feats and claims folders):
  worker : HF_TOKEN=... python tools/htstep_hub.py work --name colab-t4 --model k400_f16 \\
               --repo /content/InternVideo --video-ckpt .../1B_ft_k710_ft_k400_f16.pth \\
               --audio-ckpt .../audio_6b.pth
           claims a shard nobody is working on (<claims>/NNNN__<name>), extracts it, uploads
           <feats>/htstep_feats_NNNN.tar (one .npz per video + failed.txt), then takes the next one;
           exits when videos/DONE exists and every shard is done.
             k400_f16  : tools/pipeline/extract_features.py, k768 + a768 -> feats_k400/, claims_k400/
             stage2_f4 : tools/extract_internvideo2.py, v768 + v512 + a768 -> feats/, claims/ (the first features)
  status : python tools/htstep_hub.py status [--model k400_f16]

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

REPO = 'nguyenminh04/htstep-feats'     # features, claims, plans, annotations
VREPO = 'nguyenminh04/htstep-videos'   # video shards
PFX = 'htstep'                         # shard file prefix: <PFX>_videos_NNNN.tar
SHARD = 100
VFMT = 'videos/%s_videos_%04d.tar'     # path of video shard k in VREPO
FORMAT = 'bv*[height<=360][ext=mp4]+ba[ext=m4a]/b[height<=360]/bv*+ba/b'   # 360p keeps shards small
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# extraction model -> (features folder, claims folder, extractor script) in REPO
MODELS = {'stage2_f4': ('feats', 'claims', 'extract_internvideo2.py'),
          'k400_f16': ('feats_k400', 'claims_k400', 'pipeline/extract_features.py')}


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


def state(api, model='stage2_f4'):
    """Shards uploaded, shards finished with this model, its claims {shard: [(worker, age_h), ...] oldest
    first}, DONE flag."""
    fdir, cdir, _ = MODELS[model]
    files = set(api.list_repo_files(REPO, repo_type='dataset'))
    vfiles = files if VREPO == REPO else set(api.list_repo_files(VREPO, repo_type='dataset'))
    vpat = re.escape(VFMT).replace(re.escape('%s'), PFX).replace(re.escape('%04d'), r'(\d+)')
    shards = sorted(int(m.group(1)) for f in vfiles for m in [re.match(vpat + '$', f)] if m)
    done = {int(m.group(1)) for f in files for m in [re.match(r'%s/%s_feats_(\d+)\.tar$' % (fdir, PFX), f)] if m}
    claims = {}
    if any(f.startswith(cdir + '/') for f in files):
        now = datetime.now(timezone.utc)
        for e in api.list_repo_tree(REPO, path_in_repo=cdir, repo_type='dataset', expand=True):
            m = re.match(r'%s/(\d+)__(.+)$' % cdir, e.path)
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


def plan(a):
    """dl/plan.json + anno/htstep.json (see the module docstring).
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
    import csv
    import random
    for r in (REPO, VREPO):
        api.create_repo(r, repo_type='dataset', private=not a.public, exist_ok=True)
    d = os.path.join(ROOT, 'datasets', 'htstep', 'data')
    ann, spl = json.load(open(os.path.join(d, 'annotations.json'))), json.load(open(os.path.join(d, 'video_splits.json')))
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
    chunks = {'%04d' % (a.first + i): ids[i * SHARD:(i + 1) * SHARD] for i in range(-(-len(ids) // SHARD))}
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


BLOCK = ('not a bot', 'HTTP Error 429', 'Too Many Requests', 'rate-limit', 'rate limit', 'page needs to be reloaded')
# a throttled account gets "Video unavailable" for videos that exist (2026-10-16: 77-97% of a chunk). A few are real,
# so one is a failure, but a chunk with more of them than downloads (see UNAVAIL_BLOCK) counts as blocked
UNAVAIL = ('Video unavailable', 'This video is unavailable')
UNAVAIL_BLOCK = 10
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
    tmp = a.tmp or '/tmp/htstep_fetch_%s' % a.name
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
                res[v] = st
                if st[0] == 'failed' and any(m in st[1] for m in UNAVAIL):
                    n_un = sum(1 for s, r in res.values() if s == 'failed' and any(m in r for m in UNAVAIL))
                    n_ok = sum(1 for s, r in res.values() if s == 'ok')
                    if n_un >= UNAVAIL_BLOCK and n_un > n_ok:
                        res[v] = st = ('block', 'throttled: %d "unavailable" vs %d ok (%s)' % (n_un, n_ok, st[1]))
                if st[0] == 'block':
                    blocked.append(v)
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
    tmp = a.tmp or '/tmp/htstep_work_%s' % a.name
    os.makedirs(tmp, exist_ok=True)
    fdir, cdir, script = MODELS[a.model]
    while True:
        shards, done, claims, finished = state(api, a.model)
        free = [k for k in shards if k not in done and owner(claims, k, a.stale) in (None, a.name)]
        if not free:
            if finished and set(shards) <= done:
                log('ALL_DONE: %d shards' % len(shards)); return
            log('nothing free (%d uploaded, %d done, %d claimed); waiting' % (len(shards), len(done), len(claims)))
            time.sleep(300); continue
        k = free[0]
        api.upload_file(path_or_fileobj=('%s %d' % (a.name, time.time())).encode(), path_in_repo='%s/%04d__%s' % (cdir, k, a.name),
                        repo_id=REPO, repo_type='dataset', commit_message='claim %04d %s' % (k, a.name))
        time.sleep(5)   # two workers claiming the same shard at once: the older claim wins
        _, done, claims, _ = state(api, a.model)
        if k in done or owner(claims, k, a.stale) != a.name:
            log('shard %04d taken by %s, next' % (k, owner(claims, k, a.stale))); continue
        log('shard %04d: downloading' % k)
        src = hf_hub_download(VREPO, VFMT % (PFX, k), repo_type='dataset', local_dir=tmp,
                              token=api.token)
        vids, out = os.path.join(tmp, 'v%04d' % k), os.path.join(tmp, 'f%04d' % k)
        with tarfile.open(src) as t:
            t.extractall(vids)
        os.remove(src)
        cmd = [sys.executable, '-u', os.path.join(ROOT, 'tools', script), '--videos', vids, '--out', out,
               '--repo', a.repo, '--audio-ckpt', a.audio_ckpt]
        if a.model == 'stage2_f4':
            cmd += ['--video-ckpt', a.video_ckpt, '--workers', str(a.workers)]
        else:   # extract_features.py finds the checkpoint by its file name in --ckpt-dir; no CSV: every .mp4 of the shard
            cmd += ['--csv', '', '--model', a.model, '--ckpt-dir', os.path.dirname(a.video_ckpt), '--decode-ahead', str(a.workers)]
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
        api.upload_file(path_or_fileobj=tar, path_in_repo='%s/%s_feats_%04d.tar' % (fdir, PFX, k), repo_id=REPO,
                        repo_type='dataset', commit_message='%s %04d (%s)' % (fdir, k, a.name))
        log('shard %04d done by %s: %d videos, %d failed, %.0f min' % (k, a.name, len(npz), len(failed),
                                                                       (time.time() - t0) / 60))
        subprocess.run(['rm', '-rf', vids, out, tar])


def status(a):
    shards, done, claims, finished = state(hub(), a.model)
    log('%d shards uploaded%s, %d done with %s' % (len(shards), ' (all: DONE)' if finished else '', len(done),
                                                   a.model))
    for k in sorted(claims):
        if k not in done:
            w, age = claims[k][0]
            log('  shard %04d: %s, claimed %.1f h ago%s' % (k, w, age, ' (stale)' if age > a.stale else ''))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=('work', 'status', 'plan', 'fetch'))
    p.add_argument('--n', type=int, default=900, help='plan: train videos')
    p.add_argument('--public', action='store_true', help='plan: create the repos public')
    p.add_argument('--name', default='worker'); p.add_argument('--tmp', default='')
    p.add_argument('--model', default='k400_f16', choices=sorted(MODELS), help='work / status: features to extract')
    p.add_argument('--stale', type=float, default=0, help='hours after which a claim is dead '
                   '(default: 24 for work with k400_f16, 6 for stage2_f4, 2 for fetch)')
    p.add_argument('--repo', default=''); p.add_argument('--video-ckpt', default='')
    p.add_argument('--audio-ckpt', default=''); p.add_argument('--workers', type=int, default=4)
    p.add_argument('--first', type=int, default=1000, help='plan: number of the first chunk')
    p.add_argument('--retry', action='store_true', help='plan: dl/plan_retry.json from the fetched chunks')
    p.add_argument('--plan', default='dl/plan.json', help='fetch: the plan to work through')
    p.add_argument('--jobs', type=int, default=3, help='fetch: parallel yt-dlp processes')
    p.add_argument('--sleep', type=int, default=2, help='fetch: seconds yt-dlp waits before a download (x2 max)')
    p.add_argument('--cookies', default='', help='fetch: cookies.txt for yt-dlp (PC only)')
    p.add_argument('--ffmpeg', default='', help='fetch: ffmpeg binary when not on PATH')
    p.add_argument('--block-wait', type=int, default=30, help='fetch: minutes to wait after a YouTube block')
    p.add_argument('--reverse', action='store_true', help='fetch: take chunks from the end (fewer claim races)')
    p.add_argument('--no-done', dest='write_done', action='store_false', help='fetch: never write videos/DONE')
    a = p.parse_args()
    if not a.stale:   # a download chunk takes minutes to an hour, a k400_f16 shard up to ~10 h on a T4
        a.stale = 2.0 if a.cmd == 'fetch' else 24.0 if a.model == 'k400_f16' else 6.0
    {'work': work, 'status': status, 'plan': plan, 'fetch': fetch}[a.cmd](a)
