# Mọi tham số đều để ở đây. Muốn đổi thì sửa trực tiếp, không đọc file config hay tham số dòng lệnh.

cfg = {
    'output_folder': './ckpt/minimal',
    'init_rand_seed': 1234567891,
    'num_workers': 2,
    'batch_size': 4,

    'dataset': {
        'json_file': './data/youcookii/annotations/youcookii_annotations_trainval.json',
        # vector text InternVideo2 của mọi caption (512 chiều), key là "<video_id>#<chỉ số caption>"
        'caption_emb_file': './data/youcookii/caption_emb_iv2j.npz',
        'omni_emb_file': './data/youcookii/omni_emb_full.npz',
        # <video_id>.npz: video InternVideo2 v768 + audio BEATs a768, mỗi giây một hàng
        'feat_folder': './data/youcookii/iv2_feats',
        'default_fps': 16,
        'max_seq_len': 256,      # mọi video được nội suy về 256 bước
        'trunc_thresh': 0.5,
        'crop_ratio': [0.9, 1.0],
        'num_classes': 16,       # một video có tối đa 16 caption
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
        'omni_head_dim': 3584,
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
            'loss_weight_ground': 1.0,
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
