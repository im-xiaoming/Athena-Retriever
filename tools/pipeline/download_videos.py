"""Download the HT-Step videos listed in shards.csv from YouTube with yt-dlp.

  python tools/pipeline/download_videos.py                                   # everything as set in CONFIG
  python tools/pipeline/download_videos.py --csv tools/pipeline/shards_part1.csv   # this machine's half
  python tools/pipeline/download_videos.py --jobs 8 --cookies a.txt b.txt   # any CONFIG entry can be overridden
  python tools/pipeline/download_videos.py --chunks 4047 4048
  python tools/pipeline/download_videos.py --status all_failed --only-blocked   # redo the videos YouTube blocked

Every setting is in CONFIG below. The video ids of a chunk come from the plan files on HF
(dl/plan*.json of nguyenminh04/htstep-feats). Output: <out>/<chunk>/<id>.mp4 (360p), <out>/<chunk>/.complete when the chunk is finished, and
<out>/results.csv (chunk, id, status, reason, time). A rerun skips videos that are on disk or already
failed for good, so a stopped run just resumes.

Each job runs its own yt-dlp process. Job i uses cookies file i % n, so several accounts share the
load. Cloud machines (Colab, Kaggle) usually need no cookies; a home IP does. When YouTube blocks us
("not a bot", 429, "page needs to be reloaded") every job pauses block_wait minutes and the video is
tried again.

Needs yt-dlp, ffmpeg (merges video + audio) and, for recent YouTube, deno on PATH.
HF_TOKEN (environment or .env) reads the plan files.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

CONFIG = {
    # ---- what to download ----
    'csv': os.path.join(HERE, 'shards.csv'),   # shard table (one row per chunk)
    'status': ['todo', 'claimed', 'all_failed'],   # chunks of the CSV with these statuses (= no video shard yet)
    'chunks': [],                          # these chunk numbers instead of `status`, e.g. ['4047', '4048']
    'only_blocked': False,                 # only videos that failed earlier with "The page needs to be reloaded"
    'retry_failed': False,                 # also retry videos that failed for good in an earlier run
    # ---- where ----
    'out': 'htstep_videos',                # <out>/<chunk>/<id>.mp4 and <out>/results.csv
    # ---- how ----
    'jobs': 4,                             # parallel yt-dlp processes
    'cookies': [],                         # cookies.txt files (Netscape format); job i uses file i % n
    'format': 'bv*[height<=360][ext=mp4]+ba[ext=m4a]/b[height<=360]/bv*+ba/b',   # 360p, keeps files small
    'sleep': 2,                            # seconds yt-dlp waits before each download (random, up to 2x)
    'block_wait': 30,                      # minutes every job pauses after YouTube blocks us
    'block_tries': 3,                      # times one video is tried again after blocks
}

REPO = 'nguyenminh04/htstep-feats'
PLANS = ['dl/plan.json', 'dl/plan_retry.json', 'dl/plan_full.json', 'dl/plan_retry2.json']

BLOCK = ('not a bot', 'HTTP Error 429', 'Too Many Requests', 'rate-limit', 'page needs to be reloaded')
RETRY = ('Sign in', 'sign in', 'HTTP Error 403', 'HTTP Error 5', 'Connection', 'timed out', 'file is empty')


def hf_token():
    tok = os.environ.get('HF_TOKEN')
    env = os.path.join(ROOT, '.env')
    if not tok and os.path.exists(env):
        for line in open(env):
            if line.startswith('HF_TOKEN='):
                tok = line.strip().split('=', 1)[1]
    return tok


# ======================================= WHAT TO DOWNLOAD =======================================================
def video_list(a):
    """[(chunk, video id)] to download, from the CSV and the plan / done files on HF."""
    from huggingface_hub import hf_hub_download
    get = lambda path: json.load(open(hf_hub_download(REPO, path, repo_type='dataset', token=hf_token())))
    with open(a.csv) as f:
        rows = list(csv.DictReader(f))
    if a.chunks:
        rows = [r for r in rows if r['chunk'] in a.chunks]
    else:
        rows = [r for r in rows if r['status'] in a.status]
    plan = {}
    for p in PLANS:
        plan.update(get(p))
    todo = []
    for r in rows:
        ids = plan[r['chunk']]
        if a.only_blocked:   # only the videos that an earlier run lost to a YouTube block
            if not r['failed_page_reload'] or int(r['failed_page_reload']) == 0:
                continue
            failed = get('dl/done/%s.json' % r['chunk'])['failed']
            ids = [v for v in ids if 'page needs to be reloaded' in failed.get(v, '')]
        todo += [(r['chunk'], v) for v in ids]
    return todo


# ======================================= DOWNLOAD =======================================================
class Downloader:
    def __init__(self, a):
        self.a = a
        self.lock = threading.Lock()          # results.csv and the counters are shared by the jobs
        self.resume_at = 0.0                  # every job waits until then after a block
        self.count = {'ok': 0, 'failed': 0, 'retry': 0}
        self.left = {}                        # chunk -> videos not finished yet; at 0 the chunk gets its .complete file
        self.results = os.path.join(a.out, 'results.csv')
        new = not os.path.exists(self.results)
        self.log = open(self.results, 'a', newline='')
        self.writer = csv.writer(self.log)
        if new:
            self.writer.writerow(['chunk', 'id', 'status', 'reason', 'time'])

    def yt_dlp(self, vid, out, slot):
        cmd = ['yt-dlp', '--quiet', '--no-warnings', '--no-progress', '-f', self.a.format,
               '--merge-output-format', 'mp4', '-o', out,
               '--sleep-interval', str(self.a.sleep), '--max-sleep-interval', str(self.a.sleep * 2)]
        cookie_copy = None
        if self.a.cookies:
            # yt-dlp writes the cookies file back when it exits: give each process its own copy
            cookie_copy = tempfile.NamedTemporaryFile(suffix='.txt', delete=False).name
            shutil.copy(self.a.cookies[slot % len(self.a.cookies)], cookie_copy)
            cmd += ['--cookies', cookie_copy]
        try:
            r = subprocess.run(cmd + ['https://www.youtube.com/watch?v=' + vid], capture_output=True, text=True)
        finally:
            if cookie_copy:
                os.remove(cookie_copy)
        return r.returncode == 0 and os.path.exists(out), r.stderr

    def one(self, item, slot):
        """Download one video, retrying after blocks; returns its status."""
        chunk, vid = item
        folder = os.path.join(self.a.out, chunk)
        os.makedirs(folder, exist_ok=True)
        out = os.path.join(folder, vid + '.mp4')
        for attempt in range(self.a.block_tries + 1):
            time.sleep(max(0.0, self.resume_at - time.time()))
            ok, err = self.yt_dlp(vid, out, slot)
            for f in os.listdir(folder):   # leftovers of a failed merge (.part, .f137.mp4, ...)
                if f.startswith(vid + '.') and not (ok and f == vid + '.mp4'):
                    os.remove(os.path.join(folder, f))
            reason = (err.strip().splitlines() or [''])[-1][:200]
            if ok:
                return self.record(chunk, vid, 'ok', '')
            if not any(b in err for b in BLOCK):
                status = 'retry' if any(m in err for m in RETRY) else 'failed'
                return self.record(chunk, vid, status, reason)
            with self.lock:   # blocked: everyone pauses, then this video is tried again
                if self.resume_at < time.time():
                    print('%s BLOCKED (%s): all jobs pause %d min' % (time.strftime('%H:%M:%S'), reason[:80],
                                                                       self.a.block_wait), flush=True)
                    self.resume_at = time.time() + self.a.block_wait * 60
        return self.record(chunk, vid, 'retry', 'still blocked: ' + reason)

    def record(self, chunk, vid, status, reason):
        with self.lock:
            self.writer.writerow([chunk, vid, status, reason, time.strftime('%Y-%m-%d %H:%M:%S')])
            self.log.flush()
            self.count[status] += 1
            self.left[chunk] -= 1
            if self.left[chunk] == 0:         # extract_features.py takes a chunk only when this file exists
                open(os.path.join(self.a.out, chunk, '.complete'), 'w').close()
            n = sum(self.count.values())
            if status != 'ok' or n % 20 == 0:
                print('%s %d/%d  ok %d failed %d retry %d  | %s %s %s' % (
                    time.strftime('%H:%M:%S'), n, self.total, self.count['ok'], self.count['failed'],
                    self.count['retry'], vid, status, reason[:80]), flush=True)
        return status

    def run(self, todo):
        # skip what is on disk, and the videos that already failed for good in an earlier run
        all_items, gone = todo, set()
        if not self.a.retry_failed:
            with open(self.results) as f:
                gone = {(r['chunk'], r['id']) for r in csv.DictReader(f) if r['status'] == 'failed'}
        todo = [(c, v) for c, v in todo
                if (c, v) not in gone and not os.path.exists(os.path.join(self.a.out, c, v + '.mp4'))]
        self.total = len(todo)
        for c, v in todo:
            self.left[c] = self.left.get(c, 0) + 1
        for c in {c for c, _ in all_items} - set(self.left):   # nothing left to download: the chunk is complete
            if os.path.isdir(os.path.join(self.a.out, c)):
                open(os.path.join(self.a.out, c, '.complete'), 'w').close()
        print('%d videos to download with %d jobs%s -> %s' % (
            len(todo), self.a.jobs, ', %d cookies files' % len(self.a.cookies) if self.a.cookies else '',
            self.a.out), flush=True)
        with ThreadPoolExecutor(self.a.jobs) as pool:   # each thread runs its own yt-dlp process
            list(pool.map(self.one, todo, [i % self.a.jobs for i in range(len(todo))]))
        print('done: %s; details in %s' % (self.count, self.results), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in CONFIG.items():   # every CONFIG entry becomes an option: 'block_wait' -> --block-wait
        opt = '--' + k.replace('_', '-')
        if isinstance(v, bool):
            p.add_argument(opt, action='store_true', default=v)
        elif isinstance(v, list):
            p.add_argument(opt, nargs='*', default=v)
        else:
            p.add_argument(opt, type=type(v), default=v)
    a = p.parse_args()
    if not shutil.which('yt-dlp'):   # yt-dlp installed in this Python env but not on PATH
        os.environ['PATH'] = os.path.dirname(sys.executable) + os.pathsep + os.environ['PATH']
    os.makedirs(a.out, exist_ok=True)
    Downloader(a).run(video_list(a))
