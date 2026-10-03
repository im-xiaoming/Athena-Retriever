"""Chuẩn bị clip sự kiện YouCook2 cho OmniRetriever-7B, làm thầy cho train_event.py.

Bước 1, chạy ở máy có video gốc YouCook2 (Windows được, chỉ cần python và ffmpeg):

  python tools/omni_youcook2.py cut --videos D:/YouCookII/videos --out clips --workers 8

  Cắt mỗi đoạn GT thành một clip riêng, mã hoá lại để mốc bắt đầu chính xác
  (cắt kiểu -c copy nhảy về keyframe gần nhất, lệch tới vài giây). Hạ xuống
  360p vì OmniRetriever chỉ lấy vài khung hình mỗi clip. Chạy lại thì bỏ qua
  clip đã có.

  Đã có sẵn clip cắt bằng cut.ipynb ở data/youcook2_cut/videos, tên "<video>_<start>_<end>.mp4",
  thì bỏ qua bước này và dùng --naming range ở bước 2 (mặc định). Clip đó cắt kiểu
  -c copy nhưng đo 200 clip thì độ dài lệch tối đa 0.26 giây, dùng được.

Bước 2, tạo manifest với đường dẫn clip ở nơi sẽ chạy OmniRetriever:

  python tools/omni_youcook2.py manifest --clips /content/videos --out youcook2_omni.jsonl

  Thêm --pilot 500 để có bản chạy thử rẻ: vector text cho kho caption tập train cộng
  500 clip tập val, đủ để chạy bước 4 trả lời câu hỏi thầy có đáng dùng không.

Bước 3, trên Colab A100 với mã nguồn OmniRetriever đã sửa:

  python -m omniretriever.cli extract youcook2_omni.jsonl \\
      --base-model /content/WAVE_HOME/WAVE-7B --adapter /content/adapters/omniretriever-7b \\
      --output omni_emb.npz --modalities text av --device cuda --dtype bfloat16

  Rồi đặt omni_emb.npz vào data/youcookii/ và bật omni_emb_file trong config.

Bước 4, đo thầy mạnh tới đâu trước khi train, zero-shot trên clip GT tập val:

  python tools/omni_youcook2.py teacher --omni data/youcookii/omni_emb.npz

  So với dòng ret_sim[oracle] / CIDEr[oracle] khi chạy train_event.py --eval.
"""
import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CAPTION_EMB = os.path.join(ROOT, 'data', 'youcookii', 'caption_emb.npz')
ANNOTATIONS = os.path.join(ROOT, 'data', 'youcookii', 'annotations',
                           'youcookii_annotations_trainval.json')


def events():
    """Mọi đoạn GT mà train_event.py dùng, khoá "<video>#<i>" giống caption_emb.npz."""
    z = np.load(CAPTION_EMB, allow_pickle=True)
    with open(ANNOTATIONS) as f:
        db = json.load(f)['database']
    for key, text in zip(z['keys'], z['sentences']):
        vid, i = str(key).rsplit('#', 1)
        start, end = db[vid]['annotations'][int(i)]['segment']
        yield str(key), vid, float(start), float(end), str(text)


def clip_name(key, start, end, naming='range'):
    """'range': tên do cut.ipynb đặt, "<video>_<start>_<end>.mp4" theo giây nguyên.
    'key': tên do lệnh cut của script này đặt, "<video>_<chỉ số đoạn>.mp4"."""
    if naming == 'key':
        return key.replace('#', '_') + '.mp4'
    fmt = lambda x: '%d' % x if float(x).is_integer() else str(x)
    return '%s_%s_%s.mp4' % (key.rsplit('#', 1)[0], fmt(start), fmt(end))


def find_video(folder, vid):
    for ext in ('.mp4', '.mkv', '.webm'):
        p = os.path.join(folder, vid + ext)
        if os.path.isfile(p):
            return p
    return None


def cut(a):
    os.makedirs(a.out, exist_ok=True)
    jobs, missing = [], set()
    for key, vid, start, end, _ in events():
        dst = os.path.join(a.out, clip_name(key, start, end, 'key'))
        if os.path.isfile(dst) and os.path.getsize(dst) > 0:
            continue
        src = find_video(a.videos, vid)
        if src is None:
            missing.add(vid); continue
        jobs.append((src, dst, start, end))
    print('%d clips to cut, %d videos missing' % (len(jobs), len(missing)), flush=True)

    def run(job):
        src, dst, start, end = job
        cmd = ['ffmpeg', '-v', 'error', '-y', '-ss', '%.3f' % start, '-i', src,
               '-t', '%.3f' % max(end - start, 0.5), '-vf', 'scale=-2:360',
               '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
               '-c:a', 'aac', '-ac', '1', '-ar', '16000', dst]
        r = subprocess.run(cmd, capture_output=True)
        return dst if r.returncode else None

    failed = []
    with ThreadPoolExecutor(a.workers) as ex:
        for n, bad in enumerate(ex.map(run, jobs), 1):
            if bad:
                failed.append(bad)
            if n % 500 == 0:
                print('  %d/%d' % (n, len(jobs)), flush=True)
    with open(os.path.join(a.out, 'missing_videos.json'), 'w') as f:
        json.dump({'missing_videos': sorted(missing), 'failed_clips': failed}, f, indent=1)
    print('done: %d failed, %d videos missing (see missing_videos.json)'
          % (len(failed), len(missing)), flush=True)


def manifest(a):
    with open(ANNOTATIONS) as f:
        subset = {v: x['subset'] for v, x in json.load(f)['database'].items()}
    evs = list(events())
    pilot_val = None
    if a.pilot:
        val = [e[0] for e in evs if subset[e[1]] == 'validation']
        pilot_val = set(np.random.default_rng(0).choice(val, min(a.pilot, len(val)), replace=False))
    n = skipped = 0
    seen = set()
    with open(a.out, 'w') as f:
        for key, vid, start, end, text in evs:
            clip = os.path.join(a.clips, clip_name(key, start, end, a.naming)).replace('\\', '/')
            if a.check and not os.path.isfile(clip):
                skipped += 1; continue
            # av doc ca hinh lan tieng tu cung file video, truong audio chi can co mat
            rec = {'id': key, 'text': text, 'video': clip, 'audio': clip}
            if pilot_val is not None:
                if subset[vid] == 'training':
                    if text in seen:          # kho caption chi can moi cau mot vector
                        continue
                    seen.add(text); rec = {'id': key, 'text': text}
                elif key not in pilot_val:
                    continue
            f.write(json.dumps(rec) + '\n')
            n += 1
    print('%d records -> %s (%d skipped, clip not found)' % (n, a.out, skipped))


def teacher(a):
    """OmniRetriever tự mô tả clip GT tập val: vector av của clip so với vector text
    của kho caption tập train, chọn câu theo đồng thuận giống train_event.py."""
    import sys
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, ROOT)
    from pycocoevalcap.cider.cider import Cider
    from train_event import mbr_pick

    om = np.load(a.omni)
    have = set(om.files)
    cap = np.load(CAPTION_EMB, allow_pickle=True)
    emb = dict(zip(cap['keys'], cap['emb']))
    with open(ANNOTATIONS) as f:
        db = json.load(f)['database']
    subset = lambda key: db[str(key).rsplit('#', 1)[0]]['subset']

    pool_text, pool_raw, pool_om, seen = [], [], [], set()
    for key, _, _, _, text in events():
        if subset(key) != 'training' or text in seen or key + '__text' not in have:
            continue
        seen.add(text); pool_text.append(text)
        pool_raw.append(emb[key]); pool_om.append(om[key + '__text'])
    pool_n = F.normalize(torch.tensor(np.stack(pool_raw)).float(), dim=-1)
    pool_om = F.normalize(torch.tensor(np.stack(pool_om)).float(), dim=-1)

    sims, gts, hyp = [], {}, {}
    for key, _, _, _, text in events():
        if subset(key) != 'validation' or key + '__av' not in have:
            continue
        q = F.normalize(torch.tensor(om[key + '__av']).float(), dim=-1)
        i = mbr_pick(pool_om @ q, pool_n, a.scale)[1]
        gt = F.normalize(torch.tensor(emb[key]).float(), dim=-1)
        sims.append(float(pool_n[i] @ gt))
        gts[len(gts)] = [text]; hyp[len(hyp)] = [pool_text[i]]
    print('OmniRetriever zero-shot, %d val clips, pool %d captions' % (len(sims), len(pool_text)))
    print('ret_sim %.3f   CIDEr %.2f' % (np.mean(sims), Cider().compute_score(gts, hyp)[0] * 100))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('cut')
    c.add_argument('--videos', required=True, help='thư mục video gốc YouCook2, tên <id>.mp4')
    c.add_argument('--out', default='clips')
    c.add_argument('--workers', type=int, default=8)
    m = sub.add_parser('manifest')
    m.add_argument('--clips', required=True, help='thư mục clip, đúng đường dẫn ở máy chạy Omni')
    m.add_argument('--out', default='youcook2_omni.jsonl')
    m.add_argument('--check', action='store_true', help='bỏ record không tìm thấy clip')
    m.add_argument('--naming', choices=('range', 'key'), default='range',
                   help='range: clip của cut.ipynb; key: clip của lệnh cut ở đây')
    m.add_argument('--pilot', type=int, default=0,
                   help='chỉ lấy text kho train và N clip val ngẫu nhiên, để chạy thử')
    t = sub.add_parser('teacher')
    t.add_argument('--omni', required=True, help='file npz do omniretriever.cli extract sinh ra')
    t.add_argument('--scale', type=float, default=50.0, help='nhiệt độ khi chọn câu đồng thuận')
    a = p.parse_args()
    {'cut': cut, 'manifest': manifest, 'teacher': teacher}[a.cmd](a)
