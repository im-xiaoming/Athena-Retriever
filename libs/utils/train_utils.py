import copy
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from ..modeling import MaskedConv1D, upgrade_state_dict
from .lr_schedulers import LinearWarmupCosineAnnealingLR


def fix_random_seed(seed, include_cuda=True):
    """Seed python, numpy and torch; with include_cuda also make cuDNN / cuBLAS deterministic.
    Returns the torch generator, used by the data loader."""
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:
        # training: disable cudnn benchmark to ensure the reproducibility
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # this is needed for CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator


def make_scheduler(optimizer, optimizer_config, num_iters_per_epoch, num_epochs):
    """Linear warm-up for warmup_epochs, then cosine decay over num_epochs; steps every iteration."""
    assert optimizer_config["warmup"] and optimizer_config["schedule_type"] == "cosine", \
        "only warm-up + cosine is supported"
    warmup_steps = optimizer_config["warmup_epochs"] * num_iters_per_epoch
    max_steps = (num_epochs + optimizer_config["warmup_epochs"]) * num_iters_per_epoch
    return LinearWarmupCosineAnnealingLR(optimizer, warmup_steps, max_steps)


def build_optimizer(model, opt):
    """AdamW with three parameter groups (as ActionFormer, plus one for the caption projection):
      decay    : weights of Linear / Conv layers, with weight_decay
      no_decay : biases, LayerNorm, learnable scales and other parameters, no weight decay
      clip     : clip_proj (caption vectors -> event space), its own learning rate clip_proj_lr
    """
    decay, no_decay, clip = set(), set(), set()
    decay_types = (torch.nn.Linear, torch.nn.Conv1d, MaskedConv1D)
    for mod_name, mod in model.named_modules():
        for par_name, _ in mod.named_parameters(recurse=False):
            full_name = '%s.%s' % (mod_name, par_name) if mod_name else par_name
            if 'clip_proj' in full_name:
                clip.add(full_name)
            elif par_name.endswith('weight') and isinstance(mod, decay_types):
                decay.add(full_name)
            else:
                no_decay.add(full_name)
    pd = dict(model.named_parameters())
    assert not (pd.keys() - (decay | no_decay | clip))   # every parameter is in exactly one group
    groups = [
        {'params': [pd[n] for n in sorted(decay)], 'weight_decay': opt['weight_decay']},
        {'params': [pd[n] for n in sorted(no_decay)], 'weight_decay': 0.0},
        {'params': [pd[n] for n in sorted(clip)], 'weight_decay': 0.0, 'lr': opt['clip_proj_lr']},
    ]
    return torch.optim.AdamW([g for g in groups if g['params']], lr=opt['learning_rate'])   # no clip_proj with text_proj none


class ModelEma:
    """Exponential moving average of the weights, as ActionFormer does (train_cfg.ema_decay, e.g. 0.999):
    with it on, every evaluation and saved checkpoint uses the averaged weights."""

    def __init__(self, model, decay):
        self.module = copy.deepcopy(model)
        self.decay = decay
        self.updates = 0
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        # decay warm-up (as timm / TensorFlow): a fixed 0.999 would still hold 76% of the random init after
        # the first epoch (276 steps) and 4% at the end, so early evaluations would score an untrained model
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for e, m in zip(self.module.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)


def load_ckpt(path):
    """torch.load for our own checkpoints on any torch version.

    torch >= 2.6 defaults to weights_only=True, which rejects the metrics dict stored
    next to the weights; torch 1.11 has no weights_only argument at all. Checkpoints written on
    Colab hold numbers pickled by numpy 2, which numpy < 2 (the PC training env) reads through
    these module aliases.
    """
    if not hasattr(np, '_core'):
        sys.modules.setdefault('numpy._core', np.core)
        sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
    return torch.load(path, map_location='cpu', weights_only=False)


def load_weights(model, path):
    """Load a training checkpoint into the model (older layouts are upgraded); returns the checkpoint."""
    ck = load_ckpt(path)
    sd = upgrade_state_dict(ck['state_dict'])
    model.load_state_dict(sd)
    return ck
