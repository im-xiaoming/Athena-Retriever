# Bàn giao phiên làm việc UniAV, chiều 2026-10-04

Repo: `/home/minh/projects/UniAV`. Máy: WSL2, RTX 3060 12 GB, RAM 15.9 GB.
Bản bàn giao sáng nay: `docs/HANDOFF_2026-10-04_am.md`. Bản trước nữa: `docs/HANDOFF_2026-10-03.md`.

**Hai nhánh git:**
- `test` (commit `2bdbfa8`, đã push): trạng thái dùng cho demo Colab của `uniav_api`. **Không push gì thêm lên nhánh này.**
- `exp-iv2-variants` (đã push): mọi việc từ 13:40 trở đi. Người dùng muốn: mỗi hướng việc mới là một nhánh mới, chỉ push lên nhánh đó.

Đọc thêm khi cần chi tiết:
- `experiments/NOTES.md`: bài học từ từng thử nghiệm
- `experiments/RESULTS.md`: bảng kết quả, sinh bằng `tools/summarize_runs.py`
- `experiments/plots/`: biểu đồ, sinh bằng `tools/plot_runs.py`
- `uniav_api/README.md`: tài liệu API
- `docs/UniAV_new.drawio` (+ `.drawio.svg`), `docs/UniAV_new_walkthrough.md`: kiến trúc mới và luồng chạy qua một ví dụ

---

## 1. Trạng thái lúc viết

VARIANTS_STATUS

Không có phiên Colab nào mở. Không có tiến trình nền nào khác.

---

## 2. Việc đã làm hôm nay

1. **Trích đặc trưng InternVideo2 đủ 1500 video.**
   - Đặc trưng có ở ba nơi: trên máy `data/youcookii/iv2_feats/`, trên Drive `MyDrive/uniav_omni/iv2_feats/`, trên HF `nguyenminh04/uniav-youcook2-data/iv2_feats/` (8 tar + `manifest.json`).
   - Video `wii9jNiNl9Y` có audio ngắn hơn video, từng làm script trích chết, kéo theo 72 video sau nó không được trích. Đã sửa trong `tools/extract_internvideo2.py` (bù 0 cho audio) và thêm bảo vệ: một video lỗi không còn làm dừng cả lượt. 73 video còn thiếu được trích lại trên một phiên A100 riêng.
   - `tools/iv2_finish.sh`: script nền tự kết thúc một lượt trích trên Colab. Nó kiểm tra từng file, đồng bộ lên Drive, đẩy lên HF, rồi mới tắt phiên.
2. **Dataset nhận nhiều loại đặc trưng:** `dataset.feat_source: onepeace | iv2 | iv2+onepeace`. Với `onepeace`, đầu ra giống hệt code cũ. Đặc trưng InternVideo2 có 1 dòng mỗi giây, được chuẩn hoá L2 rồi co về 256 bước. `train_event.py` tự đặt số kênh đầu vào của model.
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

4. **`uniav_api` chuyển sang model `iv2`:**
   - Checkpoint `ckpt/api/uniav_iv2.pth`: fp16, chứa luôn cấu hình train, sinh bằng `tools/export_api_ckpt.py`. Bản trên HF: `nguyenminh04/uniav-youcook2-data/api/uniav_iv2.pth`.
   - 8 samples đã trích sẵn, hàm `uv.describe_sample`, `uv.plot`.
   - Tìm kiếm chạy được mà không cần encoder text 6 GB.
   - Notebook `uniav_api/demo_colab.ipynb`. Đã chạy thử đúng luồng của notebook trên một bản clone mới.
   - Model qua API khớp lúc train: R@0.5 54.27 bằng đúng số khi chấm lúc train. `min_score` chọn lại là 0.36.
5. **Sơ đồ kiến trúc, walkthrough, biểu đồ** (`features_*.png` so riêng các loại đặc trưng).

---

## 3. Thử nghiệm biến thể (chiều nay, nhánh `exp-iv2-variants`)

VARIANTS_RESULTS

---

## 4. Việc tiếp theo

NEXT_STEPS

---

## 5. Lưu ý vận hành mới

- **RAM là giới hạn thật.** Một lượt train dùng khoảng 6 GB, kernel Jupyter của người dùng thường chiếm 2 đến 2.5 GB. Không chạy encoder InternVideo2 (checkpoint 2.8 GB) song song với train: lần làm vậy, lượt train bị OOM-kill.
- **Hồ sơ lượt chạy bị đánh dấu `dirty`** nếu còn file đã theo dõi mà chưa commit, kể cả `experiments/RESULTS.md`. Commit trước khi khởi động hàng đợi.
- **Hàng đợi:** `tools/run_queue_iv2.sh` (so sánh đặc trưng) và `tools/run_queue_variants.sh` (biến thể). Cả hai bỏ qua các lượt đã có `final_eval`, nên chạy lại an toàn.
- **Colab:** nhớ tắt phiên khi xong. Keepalive dừng bằng `touch logs/keepalive.stop`.
- **Encoder InternVideo2 cho `describe_video`** (`uniav_api/encoders/internvideo2.py`) chưa được so với đặc trưng trích trên Colab. Lần thử dừng giữa chừng vì OOM. Trọng số đã có ở `ckpt/internvideo2/`, `timm` và `torchaudio` đã cài vào `uniav-api-env`. Cần chạy so sánh khi máy rảnh.
