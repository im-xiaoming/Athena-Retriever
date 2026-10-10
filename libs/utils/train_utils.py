import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

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
