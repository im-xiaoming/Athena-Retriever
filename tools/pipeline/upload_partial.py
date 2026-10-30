"""Upload chunks that are not finished (e.g. before shutting a machine down): tar the existing npz + failed.txt + partial.txt
(ids still missing) to feats/htstep_feats_<chunk>.tar. A later full upload of the chunk overwrites it. No .uploaded marker is written.
Usage: python upload_partial.py <videos_dir> <out_dir> [token_file]   (token from env HF_TOKEN or the file)"""
import os, sys, tarfile, tempfile
from huggingface_hub import HfApi
vid, out = sys.argv[1], sys.argv[2]
tok = os.environ.get('HF_TOKEN') or open(sys.argv[3]).read().strip()
api = HfApi(token=tok); R = 'nguyenminh04/htstep-feats-k400'
from huggingface_hub import hf_hub_download
import re
on_hf = {m.group(1): int(m.group(2)) for m in re.finditer(r'^\| (\d{4}) \| (\d+) \|', open(hf_hub_download(R, 'README.md', repo_type='dataset', token=tok)).read(), re.M)}
for chunk in sorted(os.listdir(vid)):
    d = os.path.join(vid, chunk)
    if not os.path.isdir(d) or chunk.endswith(('.tmp', '.tmp_dl')) or os.path.exists(os.path.join(out, '.uploaded', chunk)): continue
    ids = sorted(f[:-4] for f in os.listdir(d) if f.endswith('.mp4'))
    have = [i for i in ids if os.path.exists(os.path.join(out, i + '.npz'))]
    if not have or on_hf.get(chunk, 0) >= len(have): continue   # nothing to add, or HF already has as many (never overwrite a fuller tar)
    miss = [i for i in ids if i not in have]
    tmp = tempfile.mktemp(suffix='.tar')
    with tarfile.open(tmp, 'w') as t:
        for i in have: t.add(os.path.join(out, i + '.npz'), arcname=i + '.npz')
        for name, body in (('failed.txt', ''), ('partial.txt', '\n'.join(miss))):
            p = tempfile.mktemp(); open(p, 'w').write(body); t.add(p, arcname=name); os.remove(p)
    api.upload_file(path_or_fileobj=tmp, path_in_repo='feats/htstep_feats_%s.tar' % chunk, repo_id=R, repo_type='dataset',
                    commit_message='partial chunk %s (%d/%d)' % (chunk, len(have), len(ids)))
    os.remove(tmp); print(chunk, len(have), len(ids), flush=True)
