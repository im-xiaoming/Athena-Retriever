# Bàn giao cho phiên Claude trên Kaggle, 2026-10-04 tối

Từ phiên Claude chạy trên PC của người dùng (WSL, RTX 3060), tên `uniav-88`. Người dùng không muốn
train trên PC nữa: **mọi lượt train từ giờ chạy trên Kaggle.** Bối cảnh đầy đủ của ngày hôm nay nằm
trong `docs/HANDOFF.md`; file này chỉ gồm những gì cần để làm tiếp.

## 0. Quy tắc làm việc với người dùng

- **Chat bằng tiếng Việt.** Code, chú thích, log đều bằng tiếng Anh.
- **Git:** repo `https://github.com/im-xiaoming/UniAV-fixed` (public).
  - Nhánh mới nhất là `exp-dense-capgen`.
  - Mỗi hướng việc mới phải nằm trên **một nhánh mới**, tạo từ nhánh đó, và chỉ push lên nhánh mới ấy.
  - **Không bao giờ push lên `test`:** demo Colab của `uniav_api` clone từ nhánh này.
  - Commit trước khi train. `train_event.py` ghi commit vào hồ sơ lượt chạy, và đánh dấu `dirty` nếu còn thay đổi chưa commit.
- **Mọi lượt chạy phải có hồ sơ.**
  - `train_event.py` tự ghi `experiments/runs/<hostname>-<run>.json`, gồm config, lịch sử, kết quả chấm cuối.
  - Sau đó chạy `tools/summarize_runs.py` để cập nhật `experiments/RESULTS.md`, `tools/plot_runs.py` để vẽ biểu đồ, và ghi bài học vào `experiments/NOTES.md`.
  - Trên Kaggle, hostname sẽ khác `DESKTOP-8PSQBN9`, nên tên file hồ sơ cũng khác. Không sao.
- **Log và kết quả in ra phải dễ đọc:** gom thành một bảng, không bắt người dùng đối chiếu nhiều chỗ.
- **Hỏi đường dẫn dữ liệu, không tự đoán.** Mọi thứ cần cho các việc dưới đây đã nằm trên HF hoặc GitHub (mục 2).
- **Mức nhiễu giữa các seed:** khoảng ±1.5 R@0.5, ±0.003 ret_sim, ±2 CIDEr. Chênh lệch nhỏ hơn mức này thì phải chạy thêm seed mới kết luận được.

## 1. Dựng môi trường

Môi trường train trên PC: Python 3.8, torch 1.11 cu113, numpy 1.23, pyyaml, pandas, h5py, tensorboard,
pycocoevalcap; METEOR cần `java`. Trên Kaggle dùng bản torch có sẵn là được, code không phụ thuộc torch 1.11.

```bash
git clone -b exp-dense-capgen https://github.com/im-xiaoming/UniAV-fixed.git UniAV && cd UniAV
git checkout -b <nhánh-mới>
pip install pyyaml pandas h5py tensorboard pycocoevalcap huggingface_hub
cd libs/utils && python setup.py install --user && cd ../..   # NMS bằng C++, build lại khi đổi torch
```

Nếu `import nms_1d_cpu` lỗi sau khi build, thêm thư mục build vào `sys.path`, hoặc cài bằng `pip install .` trong `libs/utils`.

## 2. Dữ liệu

HF dataset `nguyenminh04/uniav-youcook2-data` là **private**: cần token, người dùng cung cấp qua Kaggle Secrets.

| Cần cho | Ở đâu | Đặt vào |
|---|---|---|
| Annotations, `caption_emb.npz` (vector ONE-PEACE text của caption) | có sẵn trong git | `data/youcookii/` |
| Đặc trưng InternVideo2 + BEATs, 1500 video, 1 dòng/giây | HF `iv2_feats/iv2_feats_00..07.tar` + `manifest.json` (khoảng 2 GB) | giải nén vào `data/youcookii/iv2_feats/` |
| Vector thầy OmniRetriever-7B (cho mọi lượt có thầy) | HF `teacher/omni_emb_full.npz` (132 MB) | `data/youcookii/omni_emb_full.npz` |
| Checkpoint của API (model `iv2`) | HF `api/uniav_iv2.pth` | `ckpt/api/uniav_iv2.pth` |
| Đặc trưng lệch nửa giây (để ghép thành 2 dòng/giây) | HF `iv2_feats_shift/` sau khi lượt trích trên Colab xong (khoảng 20:30 hôm nay) | `data/youcookii/iv2_feats_shift/` |
| Đặc trưng ONE-PEACE (chỉ để so sánh, đã bỏ) | HF `av_features/` (11 GB) | không cần nữa |

```python
from huggingface_hub import snapshot_download
snapshot_download('nguyenminh04/uniav-youcook2-data', repo_type='dataset', local_dir='hf',
                  allow_patterns=['iv2_feats/*', 'teacher/*', 'api/*'], token=TOKEN)
# then: tar -xf hf/iv2_feats/iv2_feats_0*.tar -C data/youcookii/iv2_feats ; cp hf/teacher/omni_emb_full.npz data/youcookii/
```

Kiểm tra sau khi tải: có đúng 1500 file trong `data/youcookii/iv2_feats/`. Dataset báo `Data: 1106 train / 394 val videos`.

## 3. Cách chạy

Lệnh của mọi lượt có thầy, ví dụ `iv2` (model của API):

```bash
python train_event.py configs/youcook2_event.yaml --output <run> --note '<mô tả>' \
  --set dataset.omni_emb_file=./data/youcookii/omni_emb_full.npz dataset.feat_source=iv2 [k=v ...]
```

- Lịch mặc định: 2 epoch warm-up + 10 epoch cosine, khoảng 20 phút trên RTX 3060. Khi xong, lượt chạy tự chấm checkpoint `best_cap` (`final_eval`).
- Các tuỳ chọn mới, đều mặc định tắt:

| Tuỳ chọn | Tác dụng |
|---|---|
| `dataset.caption_space=omni` | train và chọn câu trong không gian text của thầy, không dùng ONE-PEACE text |
| `model.train_cfg.omni_target=tva` | đích của thầy = vector AV của clip + vector caption |
| `model.train_cfg.loss_weight_modal=0.1` | thêm vector đoạn chỉ từ nhánh V và chỉ từ nhánh A, kéo về vector gộp (L_D của paper OmniRetriever) |
| `dataset.max_seq_len=512 model.max_seq_len=512` | 512 bước thời gian |
| `dataset.iv2_rows_per_sec=2 dataset.iv2_folder=./data/youcookii/iv2_dense` | đặc trưng 2 dòng/giây; tạo bằng `python tools/make_iv2_dense.py` sau khi có `iv2_feats_shift` |
| `--epochs 8` | lịch ngắn hơn |
| `init_rand_seed=2024` | seed thứ hai |

- `tools/run_queue_variants.sh` là mẫu hàng đợi: bỏ qua các lượt đã chấm xong, mỗi lúc chỉ chạy một lượt.

## 4. Kết quả tới giờ (val 394 video, checkpoint `best_cap`)

| Lượt | Thay đổi | R@0.5 | R@0.7 | ret_sim | top-1 | CIDEr | METEOR |
|---|---|---|---|---|---|---|---|
| `omni100` | ONE-PEACE (mốc cũ) | 49.1 | 26.4 | 0.755 | 0.739 | 86.4 | 15.09 |
| **`iv2`** / `iv2_seed2` | InternVideo2, model của API | 54.3 / 52.8 | 31.2 / 29.6 | 0.754 / 0.758 | 0.734 / 0.741 | 87.9 / 89.9 | 15.56 / 15.49 |
| `iv2_ep8` / seed2 | 8 epoch | 53.7 / 52.2 | 30.4 / 29.9 | 0.758 / 0.758 | 0.742 / 0.742 | 89.4 / 87.6 | 15.80 / 15.53 |
| `iv2_len512` | 512 bước | 53.0 | 30.2 | **0.761** | **0.745** | 91.1 | **16.14** |
| `teach_tva` | đích thầy T+V+A | 51.8 | 30.1 | 0.760 | 0.744 | **92.2** | 16.09 |
| `txt_omni` | không gian text của thầy | 52.5 | 29.0 | 0.756 | 0.732 | 87.8 | 15.80 |
| `iv2_noteach`, `iv2_reg`, `iv2_emb04`, `iv2_ep8_reg` | bỏ thầy / dropout / trọng số loss | 51.4 – 54.4 | | 0.754 – 0.758 | | 87.6 – 89.4 | |

Tách đoạn đã chững quanh 52–54 R@0.5. Phần chọn câu tăng nhẹ với `iv2_len512` và `teach_tva`, nhưng mỗi cái mới có một seed.

**Sinh câu** (`tools/capgen/`, GPT-2 với prefix từ vector đoạn của model `iv2`), so với chọn câu trong kho trên cùng các đoạn:

| Đoạn | Cách | CIDEr | METEOR | BLEU-4 |
|---|---|---|---|---|
| val GT (3031) | chọn câu (MBR) | 88.9 | 15.13 | 7.02 |
| val GT (3031) | **GPT-2 sinh câu** | **97.6** | **15.72** | **8.41** |
| đoạn model đoán (2253) | chọn câu | 86.0 | 14.91 | 6.89 |
| đoạn model đoán (2253) | **GPT-2 sinh câu** | **95.1** | **15.42** | **7.93** |

Biến thể RAG (`--rag 5`: thêm 5 câu ứng viên) và ret_sim (đo bằng encoder ONE-PEACE text) đang chạy trên PC. Phiên PC sẽ push kết quả lên `exp-dense-capgen` khi xong.

## 5. Việc tiếp theo, theo thứ tự đề xuất

1. **Seed thứ hai** cho hai hướng có triển vọng: `iv2_len512` và `teach_tva`. Sau đó thử ghép cả hai (`max_seq_len 512` + `omni_target=tva`), chạy 2 seed.
2. **Các biến thể chưa kịp chạy:** `modal_aux` (`loss_weight_modal=0.1`) và `wide` (`model.embd_dim=768 model.head_dim=768`, 299M tham số).
3. **Đặc trưng dày 2 dòng/giây:** khi có `iv2_feats_shift` trên HF, chạy `tools/make_iv2_dense.py`, rồi train với `iv2_rows_per_sec=2` và `max_seq_len 512`. Kỳ vọng nhỏ: nhãn GT chỉ chính xác tới từng giây, và 512 bước với đặc trưng nội suy cũng không giúp tách đoạn.
4. **Sinh câu:**
   - Tạo lại `segments.npz` bằng `python tools/capgen/dump_segments.py`. Script dùng `uniav_api` và `ckpt/api/uniav_iv2.pth`, cần `transformers<4.50`.
   - Thử GPT-2 medium.
   - Train trên vector đoạn của model tốt nhất mới.
   - Đưa bộ sinh câu vào `uniav_api` dưới dạng tuỳ chọn, trên nhánh mới.
   - Lưu ý: chế độ RAG mất khoảng 30 phút mỗi epoch trên RTX 3060, vì chuỗi đầu vào dài.
5. **Suy luận:** `iou_power 0.5` thêm khoảng +0.7 R@0.5 trên cùng checkpoint. Muốn áp dụng cho API thì phải chạy lại `python -m uniav_api.calibrate` để chọn lại `min_score`.

## 6. Những gì còn chạy trên PC lúc bàn giao

- `tools/capgen/train_capgen.py --name rag5`: xong thì tự chấm ret_sim cho cả `prefix` lẫn `rag5`.
- Lượt trích lệch nửa giây trên Colab (phiên `iv2shift`): script `tools/iv2_finish.sh` trên PC tự kiểm tra từng file, lưu lên Drive và HF (`iv2_feats_shift/`), rồi tắt phiên.

Phiên PC sẽ commit các kết quả này lên `exp-dense-capgen`. Pull nhánh đó để lấy.
