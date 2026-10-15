# All settings in one place. Change them here, there are no command-line arguments.

# ---------------- data ----------------
ANNO_FILE = './data/youcookii/annotations/youcookii_annotations_trainval.json'
CAPTION_FILE = './data/youcookii/caption_emb_iv2j.npz'   # InternVideo2 text vectors of every caption (512-d)
FEAT_FOLDER = './data/youcookii/iv2_feats'               # <video_id>.npz: v768 (video) + a768 (audio), 1 row / second

MAX_SEQ_LEN = 256    # every video is resampled to 256 steps
NMAX = 16            # at most 16 captions in one video
CROP_RATIO = (0.9, 1.0)
TRUNC_THRESH = 0.5

# ---------------- model ----------------
INPUT_DIM_V = 768
INPUT_DIM_A = 768
EMBD_DIM = 512
HEAD_DIM = 512
N_HEAD = 4
ARCH = (2, 3, 5)     # (#convs, #stem transformers, #pyramid transformers)
SCALE_FACTOR = 2
FPN_STRIDES = [1, 2, 4, 8, 16, 32]
REGRESSION_RANGE = [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)]
CLIP_DIM = 512       # caption vector size
DROPPATH = 0.1

# loss weights
W_REG = 1.0
W_IOU = 1.0
W_EMB = 0.2
W_SPAN = 0.2
W_GROUND = 1.0

# inference
SCORE_THRESH = 0.05
DURATION_THRESH = 0.05
MAX_SEG = 100
IOU_POWER = 0.3
NMS_SIGMA = 0.5

# ---------------- training ----------------
SEED = 1234567891
BATCH_SIZE = 4
NUM_WORKERS = 2
LR = 1e-4
CLIP_PROJ_LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 10
WARMUP_EPOCHS = 2
CLIP_GRAD = 1.0
EMA_DECAY = 0.999
OUT_DIR = './ckpt/minimal'
