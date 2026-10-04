"""Finish the InternVideo2 extraction on Colab: check outputs, sync to Drive, pack and push to HF.

Runs on the Colab VM, driven by tools/iv2_finish.sh. Each step prints '<STEP>_OK' or
'<STEP>_FAIL' as its last line so the caller can poll the log.

  python iv2_finalize_colab.py check   # load every .npz, delete broken ones, list missing videos
  python iv2_finalize_colab.py sync    # rsync to Drive, then compare names and sizes
  python iv2_finalize_colab.py pack    # tar shards + manifest, upload to the HF dataset repo
"""
import json
import os
import subprocess
import sys
import tarfile

import numpy as np

FEATS = '/content/iv2_feats'
VIDEOS = '/content/yc2_videos'
# IV2_NAME=iv2_feats_shift keeps a second extraction apart on Drive and HF
NAME = os.environ.get('IV2_NAME', 'iv2_feats')
DRIVE = '/content/drive/MyDrive/uniav_omni/' + NAME
TAR_DIR = '/content/iv2_tar'
OUT = '/content/out'
REPO = 'nguyenminh04/uniav-youcook2-data'
TOKEN_FILE = '/content/.hf_token'
SHARD_BYTES = 256 << 20
KEYS = ('v768', 'v512', 'a768')


def check():
    bad, rows = [], {}
    for f in sorted(os.listdir(FEATS)):
        if not f.endswith('.npz'):
            continue
        try:
            z = np.load(os.path.join(FEATS, f))
            n = {k: z[k].shape[0] for k in KEYS}
            ok = len(set(n.values())) == 1 and n['v768'] > 0 and all(
                np.isfinite(z[k].astype(np.float32)).all() for k in KEYS)
        except Exception as e:   # truncated or corrupt file
            ok, n = False, str(e)
        if ok:
            rows[f[:-4]] = n['v768']
        else:
            bad.append(f); os.remove(os.path.join(FEATS, f))
    videos = sorted(f[:-4] for f in os.listdir(VIDEOS) if f.endswith('.mp4'))
    missing = [v for v in videos if v not in rows]
    log = open(os.path.join(OUT, 'iv2_full.log')).read()
    skipped = [l.split()[1].rstrip(':')[:-4] for l in log.splitlines() if l.startswith('SKIP ')]
    res = {'ok': len(rows), 'bad_removed': bad, 'videos': len(videos), 'missing': missing,
           'skipped_no_frames': skipped, 'total_seconds': int(sum(rows.values()))}
    json.dump(res, open(os.path.join(OUT, 'iv2_check.json'), 'w'), indent=1)
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in res.items()}))
    print('missing (not skipped):', [v for v in missing if v not in skipped])
    # broken files must be re-extracted; videos ffmpeg could not decode are accepted as lost
    print('CHECK_OK' if not bad and set(missing) <= set(skipped) else 'CHECK_FAIL')


def sync():
    os.makedirs(DRIVE, exist_ok=True)
    subprocess.run(['rsync', '-a', '--ignore-existing', FEATS + '/', DRIVE + '/'], check=False)
    subprocess.run(['sync'], check=False)
    local = {f: os.path.getsize(os.path.join(FEATS, f)) for f in os.listdir(FEATS) if f.endswith('.npz')}
    drive = {f: os.path.getsize(os.path.join(DRIVE, f)) for f in os.listdir(DRIVE) if f.endswith('.npz')}
    diff = [f for f in local if drive.get(f) != local[f]]
    print('colab %d files, drive %d files, %d differ or missing on drive %s'
          % (len(local), len(drive), len(diff), diff[:5]))
    if diff:   # a size mismatch means an earlier copy was cut off; overwrite those
        for f in diff:
            subprocess.run(['cp', os.path.join(FEATS, f), os.path.join(DRIVE, f)], check=False)
        diff = [f for f in local if os.path.getsize(os.path.join(DRIVE, f)) != local[f]]
        print('after recopy: %d differ' % len(diff))
    print('SYNC_OK' if not diff else 'SYNC_FAIL')


def pack():
    from huggingface_hub import HfApi
    os.makedirs(TAR_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(FEATS) if f.endswith('.npz'))
    shards, cur, size = [], [], 0
    for f in files:
        cur.append(f); size += os.path.getsize(os.path.join(FEATS, f))
        if size >= SHARD_BYTES:
            shards.append(cur); cur, size = [], 0
    if cur:
        shards.append(cur)
    manifest = {}
    for i, names in enumerate(shards):
        name = '%s_%02d.tar' % (NAME, i)
        with tarfile.open(os.path.join(TAR_DIR, name), 'w') as t:
            for f in names:
                t.add(os.path.join(FEATS, f), arcname=f)
        manifest[name] = names
    with open(os.path.join(TAR_DIR, 'manifest.json'), 'w') as fh:
        json.dump({'n_files': len(files), 'shards': manifest,
                   'format': 'one .npz per video: v768, v512, a768 float16, one row per second'}, fh)
    print('%d files in %d shards' % (len(files), len(shards)), flush=True)
    api = HfApi(token=open(TOKEN_FILE).read().strip())
    api.upload_folder(repo_id=REPO, repo_type='dataset', folder_path=TAR_DIR, path_in_repo=NAME,
                      commit_message='InternVideo2 features (%s): %d videos' % (NAME, len(files)))
    remote = {f for f in api.list_repo_files(REPO, repo_type='dataset') if f.startswith(NAME + '/')}
    want = {NAME + '/' + n for n in list(manifest) + ['manifest.json']}
    print('uploaded %d/%d files' % (len(want & remote), len(want)))
    print('PACK_OK' if want <= remote else 'PACK_FAIL')


if __name__ == '__main__':
    step = sys.argv[1]
    try:
        {'check': check, 'sync': sync, 'pack': pack}[step]()
    except Exception as e:
        print('%s: %r' % (type(e).__name__, e))
        print(step.upper() + '_FAIL')
