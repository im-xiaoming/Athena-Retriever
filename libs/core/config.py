# Every parameter lives here. Edit this file to change one; there is no config file and only
# --eval / --patience on the command line.

cfg = {
    'output_folder': './ckpt/htstep',
    'init_rand_seed': 1234567891,
    'num_workers': 2,
    'batch_size': 4,             # per GPU: with torchrun on N GPUs one step sees N * batch_size videos

    'dataset': {
        'json_file': './data/htstep/annotations.json',
        # InternVideo2 text vectors (512-d) of every caption, keys "<video_id>#<caption index>"
        'caption_emb_file': './data/htstep/caption_emb_iv2j.npz',
        # OmniRetriever vectors (3584-d) of the captions, same keys; '' = no omni head
        'omni_emb_file': '',
        # <video_id>.npz: InternVideo2 K400 f16 video k768 + BEATs audio a768, one row per second
        'feat_folder': './data/htstep/k400_feats',
        'video_key': 'k768',
        'default_fps': 16,
        'max_seq_len': 256,      # every video is resampled to 256 steps
        'trunc_thresh': 0.5,
        'crop_ratio': [0.9, 1.0],
        'num_classes': 16,       # at most 16 captions per video
    },

    'model': {
        'backbone_arch': [2, 3, 5],
        'scale_factor': 2,
        'input_dim_V': 768,
        'input_dim_A': 768,
        'n_head': 4,
        'embd_kernel_size': 3,
        'embd_dim': 512,
        'embd_with_ln': True,
        'head_dim': 512,
        'omni_head_dim': 3584,   # set to 0 by train.py when omni_emb_file is ''
        'regression_range': [[0, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 10000]],
        'head_num_layers': 3,
        'head_kernel_size': 3,
        'head_with_ln': True,
        'use_abs_pe': True,
        'max_seq_len': 256,
        'nmax': 16,
        'clip_dim': 512,
        'train_cfg': {
            'loss_weight_reg': 1.0,
            'loss_weight_iou': 1.0,
            'loss_weight_emb': 0.2,
            'loss_weight_omni_emb': 0.2,
            'loss_weight_span': 0.2,
            'cls_prior_prob': 0.01,
            'dropout': 0.0,
            'droppath': 0.1,
        },
        'test_cfg': {
            'score_thresh': 0.05,
            'duration_thresh': 0.05,
            'max_seg_num': 100,
            'iou_power': 0.3,
            'nms_sigma': 0.5,
        },
    },

    'train_cfg': {
        'clip_grad_l2norm': 1.0,
        'ema_decay': 0.999,
    },

    'opt': {
        'learning_rate': 0.0001,
        'clip_proj_lr': 0.001,
        'weight_decay': 0.0001,
        'epochs': 10,
        'warmup_epochs': 2,
    },
}
