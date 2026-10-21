"""Config overrides and the run record (experiments/runs/<host>-<run>.json) of every training run."""
import json
import os
import socket
import subprocess
import sys
import time

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_DIR = os.path.join(ROOT, 'experiments', 'runs')


def apply_overrides(cfg, items):
    """--set a.b.c=value overrides the config; values are parsed as YAML (numbers, bools, lists)."""
    for it in items:
        key, val = it.split('=', 1)
        node, parts = cfg, key.split('.')
        for k in parts[:-1]:
            node = node.setdefault(k, {})
        node[parts[-1]] = yaml.safe_load(val)
        print('Override     : %s = %r' % (key, node[parts[-1]]), flush=True)


def git_state():
    """Commit hash and whether tracked source files have uncommitted changes.

    Build artefacts (*.egg-info) are ignored: installing nms_1d_cpu rewrites them.
    """
    run = lambda *c: subprocess.run(c, cwd=ROOT, capture_output=True, text=True).stdout.strip()
    try:
        changed = [l for l in run('git', 'status', '--porcelain', '--untracked-files=no').splitlines()
                   if '.egg-info/' not in l]
        return {'commit': run('git', 'rev-parse', '--short', 'HEAD'),
                'branch': run('git', 'rev-parse', '--abbrev-ref', 'HEAD'),
                'dirty': bool(changed)}
    except OSError:
        return {'commit': 'unknown', 'branch': 'unknown', 'dirty': None}


class RunRecord:
    """Everything needed to compare and reproduce a run, saved after every epoch."""

    def __init__(self, a, cfg, out_dir):
        self.path = os.path.join(RUNS_DIR, '%s-%s.json' % (socket.gethostname(), a.output))
        os.makedirs(RUNS_DIR, exist_ok=True)
        self.data = {
            'run': a.output, 'host': socket.gethostname(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
            'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'finished': None,
            'git': git_state(), 'argv': sys.argv, 'overrides': a.set, 'note': a.note,
            'epochs': a.epochs, 'out_dir': out_dir,
            'config': cfg, 'history': [], 'best': {}, 'final_eval': None,
        }
        self.save()

    @classmethod
    def load(cls, run):
        rec = cls.__new__(cls)
        rec.path = os.path.join(RUNS_DIR, '%s-%s.json' % (socket.gethostname(), run))
        with open(rec.path) as f:
            rec.data = json.load(f)
        return rec

    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.data, f, indent=1, default=str)
