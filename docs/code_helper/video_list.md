`video_list` là **list các dict**, mỗi dict là một video. Độ dài bằng batch size (4). Danh sách này do `trivial_batch_collator` tạo ra, tức là chỉ gom các item thành list chứ không ghép tensor. Việc ghép được làm trong `preprocessing` ([event_archs.py:345](libs/modeling/event_archs.py#L345)).

Mỗi dict là kết quả của `YouCook2CaptionDataset.__getitem__` ([youcook2_cap.py:202](libs/datasets/youcook2_cap.py#L202)). T = 256 (`max_seq_len`) và P = 504 là tổng số điểm trên 6 tầng pyramid.

| Khóa | Dạng | Ý nghĩa |
|---|---|---|
| `video_id`, `fps`, `duration`, `n_cap` | – | Thông tin video và số caption. |
| `feats` | `{'visual': (768, T), 'audio': (768, T)}` | Đặc trưng đã chuẩn hóa L2 và nội suy về T bước. |
| `segments` | (N, 2) | Đoạn GT theo đơn vị lưới (bước), không phải giây. |
| `labels` | (N,) | Chỉ số caption của mỗi đoạn trong video này (0..N-1), không phải id lớp toàn cục. |
| `feat_stride`, `feat_num_frames` | số | Quy đổi giữa bước lưới và giây. |
| `points` | list 6 tensor `(T_l, 4)` | Điểm của mỗi tầng: (t, tầm hồi quy dưới, tầm trên, stride). |
| `gt_cls_labels` | (P, 16) | One-hot: caption của đoạn mà mỗi điểm thuộc về (toàn 0 nếu không thuộc). |
| `gt_offsets` | (P, 2) | Khoảng cách từ điểm tới đầu và cuối đoạn, theo đơn vị stride. |
| `cap_emb`, `cap_mask`, `cap_text` | (16, 512), (16,), list 16 | Vector, mask hợp lệ và văn bản caption của các bước (đệm tới `NMAX` = 16). |
| `cap_pool_idx` | (16,) | Vị trí của mỗi caption trong pool train dùng chung (-1 là đệm). |
| `cap_av_idx` | (16,) | Chỉ số vector teacher của clip (-1 nếu không có). Chỉ có khi bật teacher. |
| `source` | số | Id dataset (0 YouCook2, 1 COIN). |
| `segments_sec` | (N, 2) | Đoạn GT theo giây, dùng để chấm điểm. |

`preprocessing` chỉ lấy `video['feats']['visual']` và `['audio']`, ghép thành `V`, `A` có dạng (B, 768, T) cùng mask (B, 1, T). Các khóa còn lại (`segments`, `gt_cls_labels`, `cap_*`...) được `losses` và `inference` đọc trực tiếp từ `video_list`.