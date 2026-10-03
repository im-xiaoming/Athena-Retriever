# Bàn giao phiên làm việc: UniAV trên YouCook2

Repo `/home/minh/projects/UniAV`. Máy WSL2, RTX 3060 12 GB, RAM 15.9 GB.

---

## 1. Môi trường

`/home/minh/uniav-env` là venv Python 3.8 dùng để train và eval.

| Gói | Phiên bản |
|---|---|
| torch | 1.11.0+cu113 |
| numpy | **1.23.5**, không nâng lên 2.x |
| nms_1d_cpu | đã biên dịch |

Kích hoạt: `source /home/minh/uniav-env/bin/activate`

**Bẫy 1.** `import nms_1d_cpu` một mình báo thiếu `libc10.so`. Phải import torch trước.

**Bẫy 2.** Repo chạy được trên torch 2.x sau khi tôi vá 5 chỗ. Chi tiết ở mục 6.

---

## 2. Hai hướng đã làm, kết quả

### Hướng A: phân loại (cũ, đã bão hoà)

Dùng `train.py` với 59 lớp do mô hình ngôn ngữ suy ra từ caption.

| Cấu hình | Đỉnh Average mAP |
|---|---|
| Sàn zero-shot (checkpoint gốc) | 0.04 |
| 281 video train | 2.74 |
| 1103 video train | **3.48** (epoch 17) |
| thêm chính quy hoá mạnh | 3.42 |

**Kết luận: trần 3.4 tới 3.5, chính quy hoá không nâng được.** Overfit bị loại khỏi danh sách nghi phạm.

Hai giả thuyết chưa kiểm chứng về nguyên nhân trần:
- Độ phân giải thời gian. mAP tụt dốc dựng đứng theo ngưỡng tIoU (7.48 ở 0.5 xuống 0.03 ở 0.95), dấu hiệu biên không đủ mịn. Config `configs/youcook2_len512.yaml` đã chuẩn bị sẵn, an toàn với checkpoint, chưa chạy.
- Chất lượng nhãn. Taxonomy ghi rõ `validation_captions_inspected: False`. Chưa ai kiểm tra nhãn validation lần nào.

### Hướng B: tách sự kiện và sinh mô tả (mới, đang chạy)

**Không còn lớp phân loại nào.** Backbone giữ nguyên, ba đầu ra thay cho một.

Kết quả 5 epoch trên RTX 3060:

| Epoch | Recall@IoU0.5 | mIoU | cos_gt | cos_nền |
|---|---|---|---|---|
| 0 | 37.4 | 41.1 | 0.39 | 0.06 |
| 4 | 46.5 | 47.0 | **0.55** | 0.06 |

`cos_gt` 0.55 so với nền 0.06 chứng minh model sinh ra vector mô tả **đúng nội dung sự kiện**. Mỗi epoch 85 giây, trọn 42 epoch khoảng một tiếng.

---

## 3. Kiến trúc hướng B

```
                         Video dài
                             │
                      Backbone (giữ nguyên)
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        ĐẦU SỰ KIỆN     ĐẦU BIÊN       ĐẦU NHÚNG
        (B, T, 1)       (B, T, 2)      (B, T, 512)
        Có sự kiện?     Dài bao nhiêu?  Nội dung gì?
              │              │              │
              └──────┬───────┘              │
                     │                      │
            Các đoạn được tách              │
            (NMS trên điểm sự kiện)         │
                     └──────────┬───────────┘
                                │
                Mỗi đoạn một vector 512 chiều
                                │
                         So cosine với
                      kho caption tập train
                                │
                      Câu mô tả gần nhất
```

Ba hàm mất mát: focal nhị phân cho đầu sự kiện, DIoU cho đầu biên, tương phản
InfoNCE cho đầu nhúng với mẫu âm là mọi caption trong lô.

**Lưu ý về loss tương phản.** Về công thức nó là cross-entropy trên tập ứng viên
nên trông giống phân loại. Nhưng đầu ra model là một vector, lúc chạy thật không
có danh sách lớp nào tồn tại. Đây đúng là cách CLIP được huấn luyện.

---

## 4. File của hướng B

| File | Vai trò |
|---|---|
| `libs/modeling/event_archs.py` | ba đầu ra, hàm mất mát, suy luận |
| `libs/datasets/youcook2_cap.py` | dataset đọc caption, nhãn là chỉ số caption trong video |
| `train_event.py` | huấn luyện một GPU, không cần distributed |
| `configs/youcook2_event.yaml` | cấu hình |
| `data/youcookii/caption_emb.npz` | 11594 vector caption, float16, **32 MB** |

Lệnh chạy:

```bash
python train_event.py configs/youcook2_event.yaml --output eventB1 --epochs 40 2>&1 | tee eventB1.log
```

**Mẹo thiết kế đáng nhớ.** Đặt `num_classes = 16` tức số caption tối đa của một
video, và cho nhãn mang chỉ số caption thay vì id lớp. Nhờ vậy `label_points`,
focal loss và DIoU loss có sẵn dùng lại được nguyên vẹn.

---

## 5. Dữ liệu

| Đường dẫn | Nội dung |
|---|---|
| `data/youcookii/av_features/` | 1500 video, 3000 file npy, 11 GB |
| `data/youcookii/annotations/youcookii_annotations_trainval.json` | caption gốc chính thức |
| `data/youcookii/annotations/youcookii.json` | 59 lớp, 1497 video |
| `data/youcookii/annotations/youcookii_51.json` | 51 lớp sau khi gộp |
| `data/youcookii/caption_emb.npz` | nhúng caption |

Dùng được 1500 video, chia 1106 train và 394 validation, tổng 11594 đoạn.
Còn 290 video có annotation nhưng **chưa trích feature**.

### Số liệu đã đo, đừng đo lại

- Đoạn dài trung vị 14 giây, video dài trung vị 291 giây, tỉ lệ 4.8%
- Mỗi video 3 tới 16 caption, trung bình 7.7
- Đoạn chồng lấn nhau chỉ 0.1%, nên `class_aware: False` là đúng
- Mất cân bằng lớp 48 lần, 22 trên 59 lớp có dưới 50 đoạn
- 13081 trên 13829 caption là duy nhất
- **Chỉ 6.9% caption validation có mặt nguyên văn trong kho train.** Đây là trần
  của mọi chỉ số đòi lấy ra đúng chuỗi ký tự, nên đừng dùng chỉ số đó
- Đặc trưng ONE-PEACE thô ở 1536 chiều **không** khớp được với prompt văn bản,
  top-1 chỉ 0.6% trong khi đoán bừa được 1.7%. Nhánh visual đã fine-tune trên
  Kinetics-400 nên trôi khỏi không gian chung với văn bản

---

## 6. Năm chỗ đã vá để chạy được trên torch 2.x

| File | Vấn đề |
|---|---|
| `libs/utils/task_utils.py:4` | `torch._six` bị xoá từ torch 2.0 |
| `libs/utils/task_utils.py:121` | `.next()` của DataLoader bị xoá |
| `libs/utils/lr_schedulers.py:7` | `_LRScheduler` đổi tên |
| `libs/utils/metrics.py:301` | `np.float` bị xoá từ numpy 1.24 |
| `train.py`, `eval.py` | torch 2.x truyền `--local-rank` gạch ngang |

Và ba sửa đổi khác trong `train.py`:
- Thêm cờ `--pretrain` để fine-tune đúng cách. Cờ `--resume` có sẵn **không dùng
  được**, nó đặt `start_epoch` bằng epoch checkpoint cộng một nên vòng lặp train
  không chạy lần nào mà cũng không báo lỗi
- Lưu model tốt nhất. Mã gốc dựng `save_states` rồi **vứt đi**, không bao giờ ghi ra đĩa
- Đường dẫn prompt đọc từ config qua khoá `prompt_file`, kèm sửa luôn lỗi
  `data/dcase` thành `data/desed`

Trong `libs/core/config.py` đổi `train_iter_gap` từ 4 xuống 1. Cơ chế gốc bóp
iteration xuống một phần tư khi task bão hoà, chỉ có nghĩa khi chạy đa nhiệm,
chạy một task thì nó chỉ làm chậm học.

---

## 7. Công cụ sinh nhúng caption

`ONEPEACE_embed_text/` chứa bản viết lại text encoder của ONE-PEACE bằng PyTorch
thuần, **không cần fairseq**, và checkpoint rút gọn chỉ nhánh text 6 GB thay vì 15 GB.

Đã kiểm chứng: tái tạo 10 lớp DESED và so với file gốc của tác giả, sai lệch
tuyệt đối lớn nhất 1.6e-07, cosine đường chéo bằng 1.000 cho cả 10 lớp.

**Cần torch 2.x** vì dùng `weights_only`, trong khi `uniav-env` là torch 1.11.
Dựng venv riêng với `torch==2.4.1+cpu` và `numpy==1.24.4`.

Nhúng 11594 caption mất 52 phút trên CPU. Script có lưu tiến độ sau mỗi 640 câu
nên dừng giữa chừng chạy tiếp được.

---

## 8. Việc tiếp theo, xếp theo thứ tự

1. **Chạy đủ 42 epoch hướng B**, một tiếng trên RTX 3060 hoặc nửa tiếng trên A100.
   Theo dõi `cos_gt` so với `cos_nền`.
2. **Trích feature cho 290 video còn lại**, pipeline nằm ở `ONEPEACE_extract_embd_code/`.
   Thêm 19% dữ liệu.
3. **Kiểm tra thủ công 50 đoạn validation** đối chiếu video thật. Nửa tiếng làm
   việc này cho biết trần thực sự nằm ở đâu.
4. **Thử `max_seq_len: 512`** nếu quay lại hướng phân loại. VRAM 11.2 GB ở batch 4,
   chạy local không nổi, A100 thì dùng batch 8 chứ đừng 16.

### Kỳ vọng thực tế

UniAV đạt 36.1 trên ActivityNet với mười nghìn video và 200 lớp khác nhau rõ rệt.
Bài này khó hơn hẳn: 1106 video, thao tác bếp na ná nhau, nhãn do mô hình ngôn
ngữ sinh và chưa kiểm chứng. Mong con số ngang ActivityNet là không thực tế.

---

## 9. Thư mục lớn còn giữ

| Đường dẫn | Dung lượng | Có cần không |
|---|---|---|
| `ONEPEACE_extract_embd_code/` | 23 GB | cần, để trích 290 video còn lại |
| `data/` | 12 GB | cần |
| `ONEPEACE_embed_text/` | 7 GB | cần, nếu muốn sinh lại nhúng caption |
| `ckpt/multi_task_anet_unav_dcase_reproduce/` | 2 GB | checkpoint UniAV gốc, khó tải lại |

Đã xoá trong phiên này: hai thư mục checkpoint cũ của hướng phân loại, bản clone
`ONE-PEACE/`, các file log, và ba file của thiết kế trung gian đã bị thay thế.
