class_prompt = 'A visual event of ' + clss + '.' 
PtTransformer = Point Transformer, một kiến trúc dùng cho Temporal Action Localization (TAL).
FPN - Feature Pyramid Network
Stem Transformer layers là các Transformer layer nằm ở phần đầu của backbone, xử lý feature ngay sau khi feature đi qua các convolution ban đầu


stride  kernel  padding  T ra   tham so
1       3       1        256    24
2       3       1        128    24
2       5       2        128    40
2       7       3        128    56
4       5       2        64     40
4       9       4        64     72
stride, padding = self.n_qx_stride, kernel_size // 2 # original code set stride = self.n_kv_stride
positional encoding?


video_id         str                dummy_001
feats.visual     (4, 6)            
                   [' -1.13', ' -1.15', ' -0.25', ' -0.43', '  0.85', '  0.69']
                   [' -0.32', ' -2.12', '  0.47', ' -0.16', '  1.44', '  0.27']
                   ... (con 2 kenh nua)
feats.audio      (4, 6)            
                   [' -0.22', ' -0.74', '  0.56', '  0.26', ' -0.17', ' -0.68']
                   ['  0.94', '  0.49', ' -0.57', '  0.92', '  1.11', '  1.29']
                   ... (con 2 kenh nua)
segments         (2, 2)             [[1.0, 3.5], [4.0, 5.5]]
labels           (2,)               [13, 33]
fps              int                16
duration         float              120.0
feat_stride      int                8
feat_num_frames  int                16