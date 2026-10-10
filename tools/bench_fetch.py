"""Download the benchmark videos listed in benchmark/*.json and upload them to a HF dataset repo in tar shards.

Resumable: a shard that already exists on the hub is skipped. Meant to run on Colab (cloud IPs need no cookies).
  python tools/bench_fetch.py --json benchmark/dcase_all.json --prefix dcase --tmp /content/bench
Ids look like <ytid> (UnAV100, whole video) or <ytid>_<start>_<end> (DCASE, a clip of the video).
"""
import argparse, json, os, shutil, subprocess, tarfile, time
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi

p = argparse.ArgumentParser()
p.add_argument('--json', required=True)
p.add_argument('--prefix', required=True)
p.add_argument('--repo', default='', help='default: nguyenminh04/<prefix>-videos')
p.add_argument('--tmp', default='/content/bench')
p.add_argument('--shard', type=int, default=100)
p.add_argument('--jobs', type=int, default=4)
p.add_argument('--cookies', default='', help='YouTube cookies file (needed on a home IP, or once a cloud IP is flagged as a bot)')
a = p.parse_args()
a.repo = a.repo or f'nguyenminh04/{a.prefix}-videos'
api = HfApi(token=os.environ['HF_TOKEN'])
api.create_repo(a.repo, repo_type='dataset', exist_ok=True)
ids = sorted(json.load(open(a.json))['database'])
have = set(api.list_repo_files(a.repo, repo_type='dataset'))


def fetch(vid, out):
    parts = vid.split('_')
    yt, sec = (vid[:11], None) if len(vid) == 11 else (vid[:11], vid[12:].split('_'))
    cmd = ['yt-dlp', '-q', '--no-warnings', '--no-playlist', '-f', 'bv*[height<=360]+ba/b[height<=360]/b',
           '--merge-output-format', 'mp4', '-o', f'{out}/{vid}.%(ext)s', '--socket-timeout', '30', '--retries', '3']
    if a.cookies:
        cmd += ['--cookies', a.cookies]
    if sec:
        cmd += ['--download-sections', f'*{sec[0]}-{sec[1]}']
    r = subprocess.run(cmd + [f'https://www.youtube.com/watch?v={yt}'], capture_output=True, text=True)
    f = f'{out}/{vid}.mp4'
    return (vid, os.path.exists(f) and os.path.getsize(f) > 0, (r.stderr or '')[-200:])


for n, s in enumerate(range(0, len(ids), a.shard)):
    name = f'{a.prefix}_{n:03d}.tar'
    if name in have:
        continue
    out = f'{a.tmp}/{a.prefix}_{n:03d}'
    shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda v: fetch(v, out), ids[s:s + a.shard]))
    ok = [v for v, g, _ in res if g]
    failed = {v: e for v, g, e in res if not g}
    json.dump({'ok': ok, 'failed': failed}, open(f'{out}/_status.json', 'w'))
    tar = f'{a.tmp}/{name}'
    with tarfile.open(tar, 'w') as t:
        for f in sorted(os.listdir(out)):
            t.add(f'{out}/{f}', arcname=f)
    for k in range(5):
        try:
            api.upload_file(path_or_fileobj=tar, path_in_repo=name, repo_id=a.repo, repo_type='dataset'); break
        except Exception as e:
            print('upload retry', e, flush=True); time.sleep(30)
    os.remove(tar); shutil.rmtree(out, ignore_errors=True)
    print(f'{time.strftime("%T")} {name}: {len(ok)} ok, {len(failed)} failed', flush=True)
api.upload_file(path_or_fileobj=a.json, path_in_repo=os.path.basename(a.json), repo_id=a.repo, repo_type='dataset')
print('DONE', a.prefix, flush=True)
