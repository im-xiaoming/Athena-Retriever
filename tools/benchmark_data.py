"""Benchmark test sets (DCASE public_eval, UnAV-100 test) for the original UniAV and for our model, through HF.

  python tools/benchmark_data.py upload             # once, from the PC holding the UniAV release files
  python tools/benchmark_data.py fetch [--uniav] [--ours]

HF repos:
  nguyenminh04/uniav-benchmark (dataset)    annotations/<set>_all.json, and the ONE-PEACE features of the UniAV
        release in onepeace/<set>/<set>_onepeace_NNNN.tar (<id>_one_peace_audio.npy, <id>_one_peace_video_finetune.npy),
        the input of the original UniAV only
  nguyenminh04/uniav-original (model)       model_epoch_006.pth.tar, the released UniAV checkpoint
  nguyenminh04/<set>-feats (dataset)        InternVideo2 + BEATs features for our model (tools/coin_hub.py work);
        videos that could not be downloaded have none
Local layout:
  benchmark/<folder>/<set>_all.json and benchmark/<folder>/onepeace_finetune_features_025s/   (folder: SETS)
  ckpt/uniav/model_epoch_006.pth.tar
  data/<set>/iv2_feats/<id>.npz
The class prompts of the UniAV checkpoint (ONE-PEACE text vectors) stay in git: benchmark/UniAV/data/.
"""
import argparse
import os
import tarfile
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETS = {'dcase': 'desed', 'unav100': 'unav100'}   # set name -> its folder in benchmark/
REPO = 'nguyenminh04/uniav-benchmark'
CKPT_REPO = 'nguyenminh04/uniav-original'
CKPT = 'model_epoch_006.pth.tar'
ONEPEACE = 'onepeace_finetune_features_025s'
SHARD_BYTES = 1 << 30


def token():
    tok = os.environ.get('HF_TOKEN')
    if not tok and os.path.exists(os.path.join(ROOT, '.env')):
        for line in open(os.path.join(ROOT, '.env')):
            if line.startswith('HF_TOKEN='):
                tok = line.strip().split('=', 1)[1]
    return tok


def anno_path(name):
    return os.path.join(ROOT, 'benchmark', SETS[name], '%s_all.json' % name)


def upload(a):
    from huggingface_hub import HfApi
    api = HfApi(token=token())
    api.create_repo(REPO, repo_type='dataset', private=True, exist_ok=True)
    api.create_repo(CKPT_REPO, repo_type='model', private=True, exist_ok=True)
    have = set(api.list_repo_files(REPO, repo_type='dataset'))
    for name in SETS:
        api.upload_file(path_or_fileobj=anno_path(name), path_in_repo='annotations/%s_all.json' % name,
                        repo_id=REPO, repo_type='dataset')
        src = os.path.join(ROOT, 'benchmark', SETS[name], ONEPEACE)
        # both files of a video go in the same shard
        vids = sorted({f.split('_one_peace_')[0] for f in os.listdir(src) if f.endswith('.npy')})
        shards, cur, size = [], [], 0
        for v in vids:
            files = ['%s_one_peace_audio.npy' % v, '%s_one_peace_video_finetune.npy' % v]
            cur += files
            size += sum(os.path.getsize(os.path.join(src, f)) for f in files)
            if size >= SHARD_BYTES:
                shards.append(cur); cur, size = [], 0
        if cur:
            shards.append(cur)
        for i, files in enumerate(shards):
            remote = 'onepeace/%s/%s_onepeace_%04d.tar' % (name, name, i)
            if remote in have:   # resumable
                continue
            with tempfile.TemporaryDirectory(dir=a.tmp) as tmp:
                t = os.path.join(tmp, os.path.basename(remote))
                with tarfile.open(t, 'w') as tf:
                    for f in files:
                        tf.add(os.path.join(src, f), arcname=f)
                api.upload_file(path_or_fileobj=t, path_in_repo=remote, repo_id=REPO, repo_type='dataset')
            print('%s: %d files' % (remote, len(files)), flush=True)
        print('%s: %d videos in %d shards' % (name, len(vids), len(shards)), flush=True)
    api.upload_file(path_or_fileobj=os.path.join(ROOT, 'ckpt', 'uniav', CKPT), path_in_repo=CKPT,
                    repo_id=CKPT_REPO, repo_type='model')
    print('uploaded %s to %s' % (CKPT, CKPT_REPO), flush=True)


def extract(tar_dir, dst, rename=None):
    """Unpack every non-empty member of the tars in tar_dir into dst (flat); empty files are failed extractions."""
    os.makedirs(dst, exist_ok=True)
    n = 0
    for t in sorted(os.listdir(tar_dir)):
        with tarfile.open(os.path.join(tar_dir, t)) as tf:
            members = [m for m in tf.getmembers() if m.name.endswith(('.npz', '.npy')) and m.size > 0]
            for m in members:
                m.name = os.path.basename(m.name)
            tf.extractall(dst, members=members)
            n += len(members)
    return n


def fetch(a):
    from huggingface_hub import hf_hub_download, snapshot_download
    tok = token()
    both = not (a.uniav or a.ours)
    for name, folder in SETS.items():
        if a.uniav or both:
            d = snapshot_download(REPO, repo_type='dataset', token=tok, local_dir=os.path.join(a.cache, 'bench') if a.cache else None,
                                  allow_patterns=['onepeace/%s/*' % name])
            n = extract(os.path.join(d, 'onepeace', name), os.path.join(ROOT, 'benchmark', folder, ONEPEACE))
            print('%s: %d ONE-PEACE files' % (name, n), flush=True)
        if a.ours or both:
            d = snapshot_download('nguyenminh04/%s-feats' % name, repo_type='dataset', token=tok,
                                  local_dir=os.path.join(a.cache, name) if a.cache else None, allow_patterns=['feats/*.tar'])
            n = extract(os.path.join(d, 'feats'), os.path.join(ROOT, 'data', name, 'iv2_feats'))
            print('%s: %d InternVideo2 feature files' % (name, n), flush=True)
    if a.uniav or both:
        os.makedirs(os.path.join(ROOT, 'ckpt', 'uniav'), exist_ok=True)
        hf_hub_download(CKPT_REPO, CKPT, token=tok, local_dir=os.path.join(ROOT, 'ckpt', 'uniav'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['upload', 'fetch'])
    p.add_argument('--uniav', action='store_true', help='fetch: only the UniAV inputs (ONE-PEACE features, checkpoint)')
    p.add_argument('--ours', action='store_true', help='fetch: only our features (InternVideo2 + BEATs)')
    p.add_argument('--cache', default='', help='fetch: download folder (default: the HF cache)')
    p.add_argument('--tmp', default=None, help='upload: folder for the shard being written')
    a = p.parse_args()
    upload(a) if a.cmd == 'upload' else fetch(a)
