"""Run the function API on YouCook2 validation videos and compare with the annotations.

  python -m athena.demo                        # every sample in athena/samples (features shipped)
  python -m athena.demo --ids 6uHoTJSLoL8 ...  # chosen samples
  python -m athena.demo --videos data/demo_videos [--cache data/demo_cache]   # raw videos, full encoders

For each video: the events the model found, each next to the real step it matches (uv.show),
then a few text searches over the processed videos.
"""
import argparse
import os
import time

import athena as uv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', default='', help='folder of raw videos (runs the encoders)')
    ap.add_argument('--ids', nargs='*', default=None, help='sample ids (default: every sample)')
    ap.add_argument('--device', default='auto'); ap.add_argument('--cache', default='')
    ap.add_argument('--width', type=int, default=110)
    ap.add_argument('--queries', nargs='*', default=['cut the onion', 'add salt and pepper', 'fry the chicken in oil',
                                                    'put the cheese on the bread', 'boil the noodles'])
    a = ap.parse_args()
    t0 = time.time()
    pipe = uv.load(device=a.device, feature_cache=a.cache)
    print('model %s (caption space %s) on %s, loaded in %.1f s' % (pipe.run, pipe.caption_space, pipe.device,
                                                               time.time() - t0), flush=True)
    if a.videos:
        jobs = [(os.path.splitext(n)[0], os.path.join(a.videos, n)) for n in sorted(os.listdir(a.videos))
                if n.endswith(('.mp4', '.mkv', '.webm', '.mov'))]
    else:
        jobs = [(v, None) for v in (a.ids or uv.samples())]
    for vid, path in jobs:
        r = uv.describe_video(path, video_id=vid) if path else uv.describe_sample(vid)
        print('\n' + '=' * a.width)
        print('encode %.0f s, model %.2f s, %s' % (r['timing']['encode_s'], r['timing']['model_s'], r['device']))
        uv.show(r, width=a.width)
    print('\n' + '=' * a.width + '\nText search over the videos above')
    for q in a.queries:
        hits = uv.search(q, top_k=3)
        print('\n  "%s"' % q)
        for h in hits:
            print('    %-12s %6.1f-%6.1f s  %-50s %.2f' % (h['video_id'], h['start'], h['end'], h['caption'][:50],
                                                       h['score']))


if __name__ == '__main__':
    main()
