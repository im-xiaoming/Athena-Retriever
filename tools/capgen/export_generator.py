"""Pack a trained caption generator for athena: fp16 GPT-2 + prefix + its training arguments.

  python tools/capgen/export_generator.py prefix     # data/youcookii/capgen/prefix/model.pt -> ckpt/api/capgen_prefix.pth
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
name = sys.argv[1]
ck = torch.load(os.path.join(ROOT, 'data', 'youcookii', 'capgen', name, 'model.pt'), map_location='cpu', weights_only=False)
half = lambda sd: {k: v.half() if v.is_floating_point() else v for k, v in sd.items()}
gpt = {k: v for k, v in half(ck['gpt']).items() if k != 'lm_head.weight'}   # tied to the token embedding
out = os.path.join(ROOT, 'ckpt', 'api', 'capgen_%s.pth' % name)
torch.save({'prefix': half(ck['prefix']), 'gpt': gpt, 'args': ck['args']}, out)
print('%s: %.0f MB, rag %d' % (out, os.path.getsize(out) / 1e6, ck['args'].get('rag', 0)))
