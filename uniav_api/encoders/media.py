"""ffmpeg location and media probing for the InternVideo2 encoder (decoding is in tools/extract_internvideo2.py).

ffmpeg comes from PATH or from the imageio-ffmpeg wheel, so Windows, macOS and Linux work
without a separate install.
"""
import re
import shutil
import subprocess


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
