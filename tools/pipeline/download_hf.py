"""Download the HT-Step video shards listed in shards.csv from HF and unpack them, one folder per chunk.

  python tools/pipeline/download_hf.py                                   # everything as set in CONFIG
  python tools/pipeline/download_hf.py --csv tools/pipeline/shards_part1.csv   # this machine's half
  python tools/pipeline/download_hf.py --chunks 1000 1001

Every setting is in CONFIG below. A CSV row with a video_shard (videos/htstep_videos_CCCC.tar in
nguyenminh04/htstep-videos) is downloaded to <out>/<chunk>/<id>.mp4, the same layout as
download_videos.py, so extract_features.py reads both. A chunk is unpacked into <chunk>.tmp and
renamed when complete: a chunk folder on disk is whole, and a rerun skips it.

Needs huggingface_hub; HF_TOKEN in the environment or .env (the repo is private).
About 1.6 GB per chunk, 180 GB for all 113 shards.
"""
import argparse
import csv
import os
import shutil
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

CONFIG = {
    # ---- what to download ----
    'csv': os.path.join(HERE, 'shards.csv'),   # shard table (one row per chunk); rows without video_shard are skipped
    'chunks': [],                          # these chunk numbers only, e.g. ['1000', '1001']; [] = every row
    # ---- where ----
    'out': 'htstep_videos',                # <out>/<chunk>/<id>.mp4
    # ---- how ----
    'jobs': 2,                             # shards downloaded at the same time
    'max_ahead': 0,                        # pause while this many chunks wait on disk for extraction (small disks); 0 = no limit
}

VREPO = 'nguyenminh04/htstep-videos'


def hf_token():
    tok = os.environ.get('HF_TOKEN')
    env = os.path.join(ROOT, '.env')
    if not tok and os.path.exists(env):
        for line in open(env):
            if line.startswith('HF_TOKEN='):
                tok = line.strip().split('=', 1)[1]
    assert tok, 'HF_TOKEN not set'
    return tok


def waiting(a):
    """Chunk folders on disk that the extractor has not deleted yet."""
    return sum(os.path.exists(os.path.join(a.out, d, '.complete')) for d in os.listdir(a.out))


def one(row, a):
    from huggingface_hub import hf_hub_download
    while a.max_ahead and waiting(a) >= a.max_ahead:   # extract_features.py --delete-videos frees the disk
        time.sleep(30)
    chunk, folder = row['chunk'], os.path.join(a.out, row['chunk'])
    tmp = folder + '.tmp'
    t0 = time.time()
    tar = hf_hub_download(VREPO, row['video_shard'], repo_type='dataset', local_dir=tmp + '_dl', token=hf_token())
    shutil.rmtree(tmp, ignore_errors=True)
    with tarfile.open(tar) as t:
        t.extractall(tmp)
    shutil.rmtree(tmp + '_dl')
    open(os.path.join(tmp, '.complete'), 'w').close()   # extract_features.py takes the chunk only with this file
    os.rename(tmp, folder)   # only now does the chunk count as downloaded
    print('%s chunk %s: %d videos, %.1f min' % (time.strftime('%H:%M:%S'), chunk, len(os.listdir(folder)) - 1,
                                                (time.time() - t0) / 60), flush=True)


def main(a):
    with open(a.csv) as f:
        rows = [r for r in csv.DictReader(f) if r['video_shard']]
    if a.chunks:
        rows = [r for r in rows if r['chunk'] in a.chunks]
    todo = [r for r in rows if not os.path.exists(os.path.join(a.out, r['chunk'], '.complete'))]
    gb = sum(float(r['video_gb'] or 0) for r in todo)
    print('%d shards to download (%.0f GB), %d already on disk -> %s' % (len(todo), gb, len(rows) - len(todo), a.out),
          flush=True)
    os.makedirs(a.out, exist_ok=True)
    with ThreadPoolExecutor(a.jobs) as pool:
        for job in [pool.submit(one, r, a) for r in todo]:
            job.result()   # raises the first error (network, disk full) instead of hiding it
    print('done', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in CONFIG.items():   # every CONFIG entry becomes an option: 'jobs' -> --jobs
        if isinstance(v, list):
            p.add_argument('--' + k.replace('_', '-'), nargs='*', default=v)
        else:
            p.add_argument('--' + k.replace('_', '-'), type=type(v), default=v)
    main(p.parse_args())
