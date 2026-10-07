"""Quick end-to-end test on raw videos (runs the encoders).

  python test.py                                 # every video in data/demo_videos
  python test.py a.mp4 b.mp4 --text "cut the onion"
  python test.py --videos my_folder --text "add salt" "fry the chicken"

1. video only          -> events with captions
2. video + sentence(s) -> where the sentence happens in each video (only with --text)
3. sentence only       -> best segments among the videos processed above (only with --text)
"""
import argparse
import os

import athena as uv

EXT = ('.mp4', '.mkv', '.webm', '.mov')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='*', help='video files (default: every video of --videos)')
    ap.add_argument('--videos', default='data/demo_videos')
    ap.add_argument('--text', nargs='*', default=[], help='sentences for modes 2 and 3')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--cache', default='', help='folder to keep the encoder features (skips re-encoding)')
    a = ap.parse_args()
    paths = a.files or [os.path.join(a.videos, n) for n in sorted(os.listdir(a.videos)) if n.endswith(EXT)]
    uv.load(device=a.device, feature_cache=a.cache)
    for p in paths:
        r = uv.describe_video(p, video_id=os.path.splitext(os.path.basename(p))[0])
        print('\n' + '=' * 100 + '\n%s: encode %.0f s, model %.2f s' % (p, r['timing']['encode_s'], r['timing']['model_s']))
        uv.show(r)
        for t in a.text:
            print('\n  grounding "%s"' % t)
            for m in uv.ground(r['video_id'], t)['matches']:
                print('    %6.1f-%6.1f s  %.2f  %s' % (m['start'], m['end'], m['score'], m['caption'][:60]))
    for t in a.text:
        print('\nsearch "%s"' % t)
        for h in uv.search(t, top_k=3):
            print('    %-12s %6.1f-%6.1f s  %.2f  %s' % (h['video_id'], h['start'], h['end'], h['score'], h['caption'][:50]))


if __name__ == '__main__':
    main()


# python test.py data/demo_videos/-Ju39A-G0Dk.mp4 --text "cut the onion"
# /home/minh/uniav-api-env/bin/python test.py data/demo_videos/6uHoTJSLoL8.mp4