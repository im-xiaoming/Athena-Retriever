"""Video/audio decoding, copied from ONEPEACE_extract_embd_code/One_Peace_FE/video_io.py.

The decoding must match how the training features were extracted, step by step:
  video: ffmpeg -> 16 fps at native resolution -> cv2 short-side resize to 256 (INTER_LINEAR)
         -> 256x256 centre crop (the mmaction2 test pipeline of ONE-PEACE K400)
  audio: ffmpeg -> s16le stereo at the native sample rate -> mean of channels
         -> librosa.resample(kaiser_best) to 16 kHz  (= librosa.load(sr=16000) of ONE-PEACE)
ffmpeg comes from PATH or from the imageio-ffmpeg wheel, so Windows, macOS and Linux work
without a separate install.
"""
import re
import shutil
import subprocess
import tempfile

import numpy as np


def find_ffmpeg():
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    try:
        import imageio_ffmpeg
    except ImportError:
        raise RuntimeError('ffmpeg not found: install it or `pip install imageio-ffmpeg`') from None
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe(path, ffmpeg=None):
    """duration, width, height, sample_rate from `ffmpeg -i` (no ffprobe needed)."""
    ffmpeg = ffmpeg or find_ffmpeg()
    err = subprocess.run([ffmpeg, '-hide_banner', '-noautorotate', '-i', path], capture_output=True,
                         text=True, encoding='utf-8', errors='ignore').stderr
    info = {'duration': None, 'width': None, 'height': None, 'sample_rate': None}
    m = re.search(r'Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)', err)
    if m:
        h, mi, s = m.groups()
        info['duration'] = int(h) * 3600 + int(mi) * 60 + float(s)
    m = re.search(r'Stream #.*?Video:.*?\s(\d{2,5})x(\d{2,5})[\s,\[]', err)
    if m:
        info['width'], info['height'] = int(m.group(1)), int(m.group(2))
    m = re.search(r'Stream #.*?Audio:.*?(\d+) Hz', err)
    if m:
        info['sample_rate'] = int(m.group(1))
    return info


def _ffmpeg_stream(cmd, chunk_bytes):
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, bufsize=chunk_bytes * 4)
        try:
            while True:
                buf = proc.stdout.read(chunk_bytes)
                if len(buf) < chunk_bytes:
                    break
                yield buf
        finally:
            proc.stdout.close()
            ret = proc.wait()
        if ret != 0:
            err.seek(0)
            raise RuntimeError('ffmpeg failed (%d): %s' % (ret, err.read().decode('utf-8', 'ignore')[-800:]))


def resize_center_crop(img, size=256):
    """mmaction2 Resize(scale=(-1, size)) with cv2 INTER_LINEAR, then CenterCrop(size)."""
    import cv2
    h, w = img.shape[:2]
    scale = size / min(h, w)
    new_w, new_h = int(w * scale + 0.5), int(h * scale + 0.5)
    if (new_w, new_h) != (w, h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    left, top = (new_w - size) // 2, (new_h - size) // 2
    return np.ascontiguousarray(img[top:top + size, left:left + size])


def iter_video_frames(path, fps=16, size=256, ffmpeg=None, info=None):
    """RGB uint8 frames (size, size, 3) at `fps`; no auto-rotation (like decord in mmaction2)."""
    ffmpeg = ffmpeg or find_ffmpeg()
    info = info or probe(path, ffmpeg)
    w, h = info['width'], info['height']
    if not w or not h:
        raise RuntimeError('cannot read the video size of %s' % path)
    cmd = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-noautorotate', '-i', path,
           '-an', '-sn', '-vf', 'fps=%d' % fps, '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1']
    for buf in _ffmpeg_stream(cmd, w * h * 3):
        yield resize_center_crop(np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3), size)


def iter_clip_chunks(frames, num_frames, stride, clips_per_chunk):
    """Groups of consecutive frames covering up to clips_per_chunk sliding windows.

    Yields (chunk, m): chunk holds (m - 1) * stride + num_frames frames and window k is
    chunk[k*stride : k*stride + num_frames]. A video shorter than num_frames yields one
    window padded with its last frame. Same windows as the training extraction.
    """
    buf, buf_start, first, n_frames = [], 0, 0, 0

    def take(m):
        start = first * stride - buf_start
        return np.stack(buf[start:start + (m - 1) * stride + num_frames])

    for idx, frame in enumerate(frames):
        n_frames = idx + 1
        buf.append(frame)
        m = clips_per_chunk
        if idx == (first + m - 1) * stride + num_frames - 1:
            yield take(m), m
            first += m
            drop = min(first * stride - buf_start, len(buf))
            del buf[:drop]
            buf_start += drop
    if n_frames == 0:
        return
    if n_frames < num_frames:
        if first == 0:
            yield np.stack(buf + [buf[-1]] * (num_frames - len(buf))), 1
        return
    m = (n_frames - num_frames) // stride + 1 - first
    if m > 0:
        yield take(m), m


def load_audio(path, sr=16000, ffmpeg=None, res_type='kaiser_best'):
    """Mono float32 waveform at sr, as librosa.load(path, sr=16000); None if there is no audio."""
    ffmpeg = ffmpeg or find_ffmpeg()
    native_sr = probe(path, ffmpeg)['sample_rate']
    if native_sr is None:
        return None
    cmd = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-i', path, '-vn', '-sn',
           '-ac', '2', '-f', 's16le', 'pipe:1']
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or len(proc.stdout) == 0:
        return None
    wav = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    wav = wav[:len(wav) // 2 * 2].reshape(-1, 2).mean(axis=1)
    if native_sr != sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=native_sr, target_sr=sr, res_type=res_type)
    return np.ascontiguousarray(wav, dtype=np.float32)


def num_windows(total, window, hop):
    return max(0, (total - window) // hop) + 1
