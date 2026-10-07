# Bàn giao phiên làm việc UniAV, chiều 2026-10-04

Repo: `/home/minh/projects/UniAV`. Máy: WSL2, RTX 3060 12 GB, RAM 15.9 GB.
Bản bàn giao sáng nay: `docs/HANDOFF_2026-10-04_am.md`. Bản trước nữa: `docs/HANDOFF_2026-10-03.md`.

**Hai nhánh git:**
- `test` (commit `2bdbfa8`, đã push): trạng thái dùng cho demo Colab của `athena`. **Không push gì thêm lên nhánh này.**
- `exp-iv2-variants` (đã push): mọi việc từ 13:40 trở đi. Người dùng muốn: mỗi hướng việc mới là một nhánh mới, chỉ push lên nhánh đó.

Đọc thêm khi cần chi tiết:
- `experiments/NOTES.md`: bài học từ từng thử nghiệm
- `experiments/RESULTS.md`: bảng kết quả, sinh bằng `tools/summarize_runs.py`
- `experiments/plots/`: biểu đồ, sinh bằng `tools/plot_runs.py`
- `athena/README.md`: tài liệu API
- `docs/architecture.drawio` (+ `.drawio.svg`), `docs/architecture_walkthrough.md`: kiến trúc mới và luồng chạy qua một ví dụ

---

## 1. Trạng thái lúc viết

Không còn gì chạy. Hàng đợi biến thể (`tools/run_queue_variants.sh`) đã xong lúc 15:28. Kết quả ở mục 3.

Không có phiên Colab nào mở. Không có tiến trình nền nào khác.

---

## 2. Việc đã làm hôm nay

1. **Trích đặc trưng InternVideo2 đủ 1500 video.**
   - Đặc trưng có ở ba nơi: trên máy `data/youcookii/iv2_feats/`, trên Drive `MyDrive/uniav_omni/iv2_feats/`, trên HF `nguyenminh04/uniav-youcook2-data/iv2_feats/` (8 tar + `manifest.json`).
   - Video `wii9jNiNl9Y` có audio ngắn hơn video, từng làm script trích chết, kéo theo 72 video sau nó không được trích. Đã sửa trong `tools/extract_internvideo2.py` (bù 0 cho audio) và thêm bảo vệ: một video lỗi không còn làm dừng cả lượt. 73 video còn thiếu được trích lại trên một phiên A100 riêng.
   - `tools/iv2_finish.sh`: script nền tự kết thúc một lượt trích trên Colab. Nó kiểm tra từng file, đồng bộ lên Drive, đẩy lên HF, rồi mới tắt phiên.
2. **Dataset nhận nhiều loại đặc trưng:** `dataset.feat_source: onepeace | iv2 | iv2+onepeace`. Với `onepeace`, đầu ra giống hệt code cũ. Đặc trưng InternVideo2 có 1 dòng mỗi giây, được chuẩn hoá L2 rồi co về 256 bước. `train.py` tự đặt số kênh đầu vào của model.
3. **So sánh đặc trưng** (cùng thầy, đủ dữ liệu, 394 video val):

| Lượt | R@0.5 | R@0.7 | ret_sim | top-1 | CIDEr | METEOR |
|---|---|---|---|---|---|---|
| `omni100` (ONE-PEACE) | 49.13 | 26.36 | 0.755 | 0.739 | 86.4 | 15.09 |
| `omni100_seed2` (ONE-PEACE) | 47.44 | 24.48 | 0.754 | 0.736 | 85.2 | 15.44 |
| **`iv2`** (InternVideo2 v768 + BEATs a768) | **54.27** | **31.24** | 0.754 | 0.734 | 87.9 | 15.56 |
| `iv2_seed2` | 52.82 | 29.56 | 0.758 | 0.741 | 89.9 | 15.50 |
| `iv2_v512` (thêm v512) | 52.92 | 29.10 | 0.755 | 0.738 | 88.1 | 15.55 |
| `iv2op` (ghép với ONE-PEACE) | 48.83 | 25.11 | 0.756 | 0.738 | 89.1 | 15.62 |

   Kết luận:
   - **Tách đoạn tăng rõ:** R@0.5 hơn khoảng 4 đến 5 điểm, vượt mức nhiễu.
   - **Chọn câu ngang bằng.**
   - **Ghép thêm ONE-PEACE làm hỏng tách đoạn.** ONE-PEACE đã được bỏ, như người dùng muốn.

4. **`athena` chuyển sang model `iv2`:**
   - Checkpoint `ckpt/api/uniav_iv2.pth`: fp16, chứa luôn cấu hình train, sinh bằng `tools/export_api_ckpt.py`. Bản trên HF: `nguyenminh04/uniav-youcook2-data/api/uniav_iv2.pth`.
   - 8 samples đã trích sẵn, hàm `uv.describe_sample`, `uv.plot`.
   - Tìm kiếm chạy được mà không cần encoder text 6 GB.
   - Notebook `athena/demo_colab.ipynb`. Đã chạy thử đúng luồng của notebook trên một bản clone mới.
   - Model qua API khớp lúc train: R@0.5 54.27 bằng đúng số khi chấm lúc train. `min_score` chọn lại là 0.36.
5. **Sơ đồ kiến trúc, walkthrough, biểu đồ** (`features_*.png` so riêng các loại đặc trưng).

---

## 3. Thử nghiệm biến thể (chiều nay, nhánh `exp-iv2-variants`)

Mọi lượt dùng thầy 100% và đủ dữ liệu (1106 train / 394 val), trừ `iv2_noteach`. Chấm checkpoint `best_cap`.

| Lượt | Thay đổi so với `iv2` | R@0.5 | R@0.7 | ret_sim | top-1 | CIDEr | METEOR |
|---|---|---|---|---|---|---|---|
| `iv2` / `iv2_seed2` | (mốc) | 54.27 / 52.82 | 31.24 / 29.56 | 0.754 / 0.758 | 0.734 / 0.741 | 87.9 / 89.9 | 15.56 / 15.49 |
| `iv2_noteach` | bỏ thầy | 51.44 | 30.12 | 0.758 | 0.734 | 89.3 | 15.35 |
| `iv2_reg` | dropout 0.1, drop-path 0.2 | 52.23 | 29.59 | 0.757 | 0.743 | 89.4 | 15.75 |
| `iv2_emb04` | trọng số L_emb, L_span 0.4 | 53.35 | 31.34 | 0.755 | 0.734 | 89.1 | 15.57 |
| `iv2_ep8` / `iv2_ep8_seed2` | 8 epoch thay vì 10 | 53.68 / 52.19 | 30.39 / 29.89 | 0.758 / 0.758 | 0.742 / 0.742 | 89.4 / 87.6 | 15.80 / 15.53 |
| `iv2_ep8_reg` | 8 epoch + dropout | 54.44 | 31.51 | 0.754 | 0.736 | 87.6 | 15.40 |

Quét tham số suy luận trên checkpoint `iv2` (không train lại, log `logs/iv2_infer_sweep.log`):
- `iou_power` 0.5 thay vì 0.3: R@0.5 54.27 → 54.97.
- soft-NMS sigma 0.9: R@0.7 +0.7.

Chưa áp dụng vào API, vì `min_score` 0.36 đang được hiệu chỉnh theo `iou_power` 0.3.

**Kết luận:**
- Mọi biến thể đều nằm trong mức nhiễu giữa các seed (±1.5 R@0.5, ±0.003 ret_sim). Với các siêu tham số này, model đã chững lại.
- `iv2` vẫn là model của API. Train 8 epoch cho kết quả tương đương và nhanh hơn 20%.
- Thầy vẫn giữ: rẻ, và có vẻ giúp tách đoạn một chút, dù chưa ra khỏi mức nhiễu.

Biểu đồ: `experiments/plots/variants_{curves,final,losses}.png`.

---

## 4. Việc tiếp theo

Theo thứ tự nên làm:

1. ~~Kiểm tra encoder InternVideo2 của API~~: **đã xong lúc 15:40**, cosine 1.00000 so với đặc trưng Colab; một video 182 giây mất 42 giây trên RTX 3060 (mục "Checks done" trong `athena/README.md`).
2. **Có thể áp dụng `iou_power 0.5`** cho API: sửa `test_cfg.iou_power` trong config của checkpoint export, chạy lại `python -m athena.calibrate`, đặt lại `min_score`. Làm trên nhánh mới, không đụng `test`.
3. **Muốn tiến thêm cần thay đổi lớn hơn siêu tham số:**
   - **Phần chọn câu:** "trần" hiện tại là chọn câu trong kho với đoạn GT (ret_sim oracle khoảng 0.753). Cao hơn nữa cần sinh câu thay vì chọn: dùng InternVideo3-8B hoặc model ngôn ngữ để viết caption cho từng đoạn. Hoặc thêm vector text của InternVideo2 (đã căn chỉnh với v512) làm không gian chọn câu thứ hai.
   - **Phần tách đoạn:** thử `max_seq_len 512`, vì đặc trưng 1 dòng/giây nên video dài bị nén nhiều. Hoặc trích InternVideo2 dày hơn (2 dòng/giây).
   - **Thầy:** thử MLP nhỏ trên vector có sẵn, hoặc LoRA cho OmniRetriever (đã bàn từ sáng).

---

## 5. Lưu ý vận hành mới

- **RAM là giới hạn thật.** Một lượt train dùng khoảng 6 GB, kernel Jupyter của người dùng thường chiếm 2 đến 2.5 GB. Không chạy encoder InternVideo2 (checkpoint 2.8 GB) song song với train: lần làm vậy, lượt train bị OOM-kill.
- **Hồ sơ lượt chạy bị đánh dấu `dirty`** nếu còn file đã theo dõi mà chưa commit, kể cả `experiments/RESULTS.md`. Commit trước khi khởi động hàng đợi.
- **Hàng đợi:** `tools/run_queue_iv2.sh` (so sánh đặc trưng) và `tools/run_queue_variants.sh` (biến thể). Cả hai bỏ qua các lượt đã có `final_eval`, nên chạy lại an toàn.
- **Colab:** nhớ tắt phiên khi xong. Keepalive dừng bằng `touch logs/keepalive.stop`.
- **Encoder InternVideo2 cho `describe_video`** đã được kiểm tra trên máy, khớp tuyệt đối với đặc trưng Colab. Trọng số ở `ckpt/internvideo2/`, repo upstream ở `InternVideo/`. Nhớ chạy khi không có lượt train nào (đỉnh RAM khoảng 7 GB).
