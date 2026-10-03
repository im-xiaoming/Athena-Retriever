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

`ONEPEACE_extract_embd_code/onepeace_text.py` là bản viết lại text encoder của ONE-PEACE
bằng PyTorch thuần, **không cần fairseq**. Checkpoint rút gọn chỉ nhánh text nằm ở
`ONEPEACE_extract_embd_code/models/one-peace-text.pt`, 6 GB thay vì 15 GB.

Đã kiểm chứng: tái tạo 10 lớp DESED và so với file gốc của tác giả, sai lệch
tuyệt đối lớn nhất 1.6e-07, cosine đường chéo bằng 1.000 cho cả 10 lớp.

Chạy được trong `uniav-env` (torch 1.11, đã cài thêm `regex`), không cần venv riêng.
Đã kiểm: sai lệch so với `caption_emb.npz` 3e-5, đúng mức làm tròn float16.

Sinh lại toàn bộ: `python tools/embed_captions.py`, 52 phút trên CPU, lưu tiến độ
sau mỗi 640 câu nên dừng giữa chừng chạy tiếp được.

---

## 8. Việc tiếp theo, xếp theo thứ tự

1. **Sinh vector thầy OmniRetriever cho 11594 clip** rồi train lại với `omni_emb_file`, xem mục 10.
   Video gốc nằm ở máy Windows đã chạy cut.ipynb, cần A100 vì model 7B.
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
| `ONEPEACE_extract_embd_code/` | 18 GB | ba model text, video, audio ở `models/`, cần để trích 290 video còn lại |
| `data/` | 12 GB | cần |
| `ckpt/multi_task_anet_unav_dcase_reproduce/` | 2 GB | checkpoint UniAV gốc, khó tải lại |

Đã xoá trong phiên này: hai thư mục checkpoint cũ của hướng phân loại, bản clone
`ONE-PEACE/`, các file log, và ba file của thiết kế trung gian đã bị thay thế.

---

## 10. Phiên 2026-10-03: nâng cấp hướng B

Kết quả chạy 10 epoch trên RTX 3060, checkpoint `ckpt/sched10/best_cap.pth.tar` (epoch 7):

| Chỉ số | Bản cũ (Colab, epoch 8) | Bản mới |
|---|---|---|
| R@0.5 | 47.5 | **49.3** |
| R@0.7 | 24.6 | **26.6** |
| ret_sim | 0.669 | **0.748** |
| CIDEr | chưa đo | **88.0** (METEOR 14.4) |

ret_sim có sàn 0.46 (chọn bừa) và trần 0.896 (chọn câu tốt nhất có trong kho).

**Những thay đổi có tác dụng, đã đo riêng từng cái:**
- Chọn câu theo đồng thuận (MBR trên top 20), đổi lúc suy luận: CIDEr 71 → 88
- Loss nhúng so với toàn kho 8218 câu, nhãn mềm theo độ giống ONE-PEACE: ret_sim top-1 0.694 → 0.711
- Nhánh dự đoán IoU dùng để xếp hạng đoạn, `iou_power: 0.3`: R@0.5 +1.0, R@0.7 +1.4 trên cùng checkpoint
- Trọng số loss nhúng 0.2: ở trọng số 1.0 loss nhúng chiếm hết gradient sau clip, R@0.5 tụt 2 điểm
- Lịch 10 epoch: 40 epoch overfit từ epoch 7, R@0.5 rơi từ 50.8 xuống 42.9 ở epoch 12

**Đã thử, không có tác dụng:**
- Nạp checkpoint UniAV gốc (`--pretrain`): kém hơn ở cả hai lần thử
- `max_seq_len: 512`: ngang 256
- NMS cứng hoặc ngưỡng thấp: kém hơn soft-NMS 0.7
- Chính quy hoá mạnh (weight decay 0.05, dropout 0.1, cắt 50–100%): ngang bản thường
- Trung bình vector trên cả đoạn so với tại một mốc: ngang nhau, giữ vì cần cho thầy Omni

**Mô tả trên đoạn GT ngang trên đoạn dự đoán** (ret_sim 0.742 so với 0.748), nên nút thắt
của phần mô tả nằm ở chất lượng vector chứ không ở việc tách đoạn.

**Thầy OmniRetriever-7B, code xong, chưa có dữ liệu.** Ý fusion-as-teacher của bài
OmniRetriever: vector đoạn được chiếu sang không gian 3584 chiều, kéo về vector av của
clip GT do model 7B sinh, và so với kho caption trong không gian đó. Đã chạy thông với
vector giả. Quy trình ở `tools/omni_youcook2.py` (cut → manifest → extract trên A100 →
`teacher` để đo zero-shot), rồi bỏ comment `omni_emb_file` trong config.
Video gốc YouCook2 không có trên máy WSL này.

**Chấm checkpoint:** `python train_event.py configs/youcook2_event.yaml --eval <ckpt>`
in đủ các biến thể, kể cả `[oracle]` là mô tả trên đoạn GT.
Cần `pip install pycocoevalcap` (METEOR cần java, chỉ dùng trong `--eval`).

---

## 11. Dọn môi trường ngày 2026-10-03

`ONEPEACE_embed_text/` đã gộp vào `ONEPEACE_extract_embd_code/`, phần notebook cũ nằm ở
`text_embed/`. Ba model ONE-PEACE **không phải bản sao**: chúng dùng chung các lớp
attention, khoảng 1.5 GB mỗi file, còn lại là FFN và adapter riêng của từng modality.

Đã xoá ba môi trường Windows nằm trong WSL (`venv`, `op310`, `ONEPEACE_embed_text/venv`).
Gọi từ WSL thì Python khởi động nhưng không thấy gói nào. Danh sách gói lưu ở
`ONEPEACE_extract_embd_code/envs/` để dựng lại khi cần. Notebook `EXTRACT_local.ipynb`
vẫn trỏ tới hai môi trường đó.

Môi trường Linux còn lại duy nhất cho repo này là `/home/minh/uniav-env`.
