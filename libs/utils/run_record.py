"""Run record of every training run: experiments/runs/<host>-<start time>.json, saved after every epoch."""
import json
import os
import socket
import subprocess
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_DIR = os.path.join(ROOT, 'experiments', 'runs')


def git_state():
    """Commit, branch and whether tracked files have uncommitted changes."""
    run = lambda *c: subprocess.run(c, cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {'commit': run('git', 'rev-parse', '--short', 'HEAD'),
            'branch': run('git', 'rev-parse', '--abbrev-ref', 'HEAD'),
            'dirty': bool(run('git', 'status', '--porcelain', '--untracked-files=no'))}


class RunRecord:
    """Everything needed to compare and reproduce a run: config, git state, metrics of every epoch, best epoch."""

    def __init__(self, cfg, world_size):
        started = time.strftime('%Y%m%d-%H%M%S')
        self.path = os.path.join(RUNS_DIR, f'{socket.gethostname()}-{started}.json')
        os.makedirs(RUNS_DIR, exist_ok=True)
        self.data = {
            'host': socket.gethostname(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
            'gpus': world_size,
            'started': started, 'finished': None,
            'git': git_state(), 'argv': sys.argv,
            'config': cfg, 'history': [], 'best': None,
        }
        self.save()

    def add_epoch(self, epoch, seconds, losses, metrics, best):
        self.data['history'].append({'epoch': epoch, 'seconds': round(seconds, 1), **metrics, **losses})
        self.data['best'] = best
        self.save()

    def finish(self, **extra):
        self.data.update(extra, finished=time.strftime('%Y%m%d-%H%M%S'))
        self.save()

    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.data, f, indent=1, default=str)
