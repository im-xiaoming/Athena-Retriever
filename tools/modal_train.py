"""Train the event segmentation + captioning model on Modal (https://modal.com).

One-time setup (on any machine):
  pip install modal && modal setup
  modal secret create huggingface HF_TOKEN=<read token for the private nguyenminh04/* datasets>

  # 1. download the features into a Modal volume (CPU only, run once; ~4 GB YouCook2 + ~3 GB COIN)
  modal run tools/modal_train.py::prepare
  modal run tools/modal_train.py::prepare --no-coin          # YouCook2 only
  # 2. train; the arguments after --args go to train.py exactly as on a local machine
  modal run tools/modal_train.py::train --args "configs/youcook2_event.yaml --output run1 --epochs 10"
  modal run tools/modal_train.py::train --args "configs/youcook2_event.yaml --output run2 --set init_rand_seed=2"
  # 3. fetch the results (checkpoints in ckpt/<output>/, run records in experiments/runs/)
  modal volume get uniav-out ckpt ./modal_ckpt
  modal volume get uniav-out runs ./modal_runs

The GPU is A10G; pick another with `MODAL_GPU=A100 modal run ...`.
Use `modal run --detach ...` to keep a long run going after the terminal closes. The data volume
(`uniav-data`) and the output volume (`uniav-out`) persist between runs, so prepare is only needed once.
"""
import os
import shlex
import subprocess

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = '/root/athena'
DATA_VOL, OUT_VOL = '/vol/data', '/vol/out'
YC_REPO, COIN_REPO = 'nguyenminh04/uniav-youcook2-data', 'nguyenminh04/coin-data'

app = modal.App('athena-train')
data_vol = modal.Volume.from_name('uniav-data', create_if_missing=True)
out_vol = modal.Volume.from_name('uniav-out', create_if_missing=True)
hf_secret = modal.Secret.from_name('huggingface')

# the code plus the small files that live in git; features and teacher vectors come from HF into the data volume
SKIP = ['**/__pycache__', '**/build', '**/dist', '**/*.egg-info', '**/*.pyc']
image = (
    modal.Image.debian_slim(python_version='3.11')
    .apt_install('build-essential')
    .pip_install('torch==2.5.1', 'numpy<2', 'pyyaml', 'huggingface_hub', 'scipy')
    .add_local_file(os.path.join(ROOT, 'train.py'), APP_DIR + '/train.py', copy=True)
    .add_local_dir(os.path.join(ROOT, 'libs'), APP_DIR + '/libs', ignore=SKIP, copy=True)
    .add_local_dir(os.path.join(ROOT, 'configs'), APP_DIR + '/configs', copy=True)
    .add_local_dir(os.path.join(ROOT, 'data', 'youcookii', 'annotations'),
                   APP_DIR + '/data/youcookii/annotations', copy=True)
    .add_local_file(os.path.join(ROOT, 'data', 'youcookii', 'caption_emb_iv2.npz'),
                    APP_DIR + '/data/youcookii/caption_emb_iv2.npz', copy=True)
    .add_local_file(os.path.join(ROOT, 'data', 'youcookii', 'caption_emb_iv2j.npz'),
                    APP_DIR + '/data/youcookii/caption_emb_iv2j.npz', copy=True)
    .add_local_dir(os.path.join(ROOT, 'data', 'coin'), APP_DIR + '/data/coin', ignore=['**/iv2_feats'], copy=True)
    .add_local_file(os.path.join(ROOT, 'data', 'coin_exclude.txt'), APP_DIR + '/data/coin_exclude.txt', copy=True)
    .run_commands('cd %s/libs/utils && python setup.py install' % APP_DIR)   # nms_1d_cpu extension
)

# where the big inputs live in the data volume -> where train.py's config expects them
LINKS = {
    'youcookii/iv2_feats': 'data/youcookii/iv2_feats',
    'youcookii/omni_emb_full.npz': 'data/youcookii/omni_emb_full.npz',
    'coin/iv2_feats': 'data/coin/iv2_feats',
}


@app.function(image=image, volumes={DATA_VOL: data_vol}, secrets=[hf_secret], timeout=4 * 3600)
def prepare_data(coin: bool = True):
    import tarfile
    from huggingface_hub import snapshot_download

    tok = os.environ['HF_TOKEN']

    def untar(repo, pattern, dest):
        d = snapshot_download(repo, repo_type='dataset', allow_patterns=[pattern + '/*.tar', pattern + '/manifest.json'],
                              token=tok, local_dir='/tmp/hf_' + repo.split('/')[1])
        os.makedirs(dest, exist_ok=True)
        n = 0
        for t in sorted(os.listdir(os.path.join(d, pattern))):
            if not t.endswith('.tar'):
                continue
            with tarfile.open(os.path.join(d, pattern, t)) as tf:
                members = [m for m in tf.getmembers() if m.name.endswith('.npz')]
                tf.extractall(dest, members=members)
                n += len(members)
        print('%s: %d videos -> %s' % (repo, n, dest), flush=True)

    untar(YC_REPO, 'iv2_feats', DATA_VOL + '/youcookii/iv2_feats')
    d = snapshot_download(YC_REPO, repo_type='dataset', allow_patterns=['teacher/omni_emb_full.npz'], token=tok,
                          local_dir='/tmp/hf_teacher')
    os.makedirs(DATA_VOL + '/youcookii', exist_ok=True)
    os.replace(os.path.join(d, 'teacher', 'omni_emb_full.npz'), DATA_VOL + '/youcookii/omni_emb_full.npz')
    if coin:
        untar(COIN_REPO, 'feats', DATA_VOL + '/coin/iv2_feats')
    data_vol.commit()


@app.function(image=image, gpu=os.environ.get('MODAL_GPU', 'A10G'), volumes={DATA_VOL: data_vol, OUT_VOL: out_vol}, secrets=[hf_secret],
              timeout=24 * 3600)
def run_train(args: str):
    os.chdir(APP_DIR)
    for src, dst in LINKS.items():
        if not os.path.exists(os.path.join(DATA_VOL, src)):
            print('missing %s in the data volume (run prepare first); continuing without it' % src, flush=True)
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(DATA_VOL, src), dst)
    # results go to the output volume: ckpt/<output>/ and experiments/runs/*.json
    os.makedirs(OUT_VOL + '/ckpt', exist_ok=True)
    os.makedirs(OUT_VOL + '/runs', exist_ok=True)
    os.makedirs('experiments', exist_ok=True)
    for src, dst in ((OUT_VOL + '/ckpt', 'ckpt'), (OUT_VOL + '/runs', 'experiments/runs')):
        if not os.path.lexists(dst):
            os.symlink(src, dst)
    cmd = ['python', '-u', 'train.py'] + shlex.split(args)
    print('+ ' + ' '.join(cmd), flush=True)
    try:
        r = subprocess.run(cmd)
    finally:
        out_vol.commit()
    if r.returncode:
        raise RuntimeError('train.py exited with %d' % r.returncode)


@app.local_entrypoint()
def prepare(no_coin: bool = False):
    prepare_data.remote(coin=not no_coin)


@app.local_entrypoint()
def train(args: str = 'configs/youcook2_event.yaml --output modal1'):
    run_train.remote(args)
