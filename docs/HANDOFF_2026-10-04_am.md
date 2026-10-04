# Bàn giao phiên làm việc UniAV, ngày 2026-10-04

Repo: `/home/minh/projects/UniAV`, nhánh `test` (đã push lên `im-xiaoming/UniAV-fixed`, commit mới nhất khi viết file này là `d2de0d5`).
Máy: WSL2, RTX 3060 12 GB, RAM 15.9 GB. Bản bàn giao trước nằm ở `docs/HANDOFF_2026-10-03.md`.

Đọc thêm khi cần chi tiết:
- `experiments/NOTES.md`: bài học rút ra từ từng thử nghiệm
- `experiments/RESULTS.md`: bảng kết quả, sinh bằng `tools/summarize_runs.py`
- `uniav_api/README.md`: tài liệu API

---

## 1. Việc ĐANG CHẠY, phải xử lý đầu tiên

**Trích đặc trưng InternVideo2 trên Colab, phiên tên `omni` (A100).**

| Mục | Trạng thái lúc 07:15 |
|---|---|
| Tiến độ | 597/1500 video, không video nào lỗi |
| Tốc độ | khoảng 32 giây video mỗi giây |
| Dự kiến xong | khoảng 09:30 |
| Compute unit | còn 259 unit, tiêu 5.3 unit/giờ |
| Đầu ra trên Colab | `/content/iv2_feats/<video_id>.npz` |
| Sao lưu sang Drive | `MyDrive/uniav_omni/iv2_feats/`, 10 phút một lần (561 file lúc 07:15) |
| Log | `/content/out/iv2_full.log` (dòng cuối có ETA) |
| Cờ báo xong | `/content/out/iv2_full.done` |

Mỗi file `.npz` có ba mảng float16, mỗi giây video một dòng:
- `v768`: vector video gộp của encoder InternVideo2-1B, trước khi chiếu
- `v512`: vector video đã qua `vision_proj`, căn chỉnh với văn bản, chuẩn hoá L2
- `a768`: vector audio BEATs

**Giữ phiên Colab sống.** Colab tắt phiên khoảng 25 đến 30 phút sau lệnh `colab exec` cuối cùng từ máy khách. Kernel bận không được tính là hoạt động.
- Vòng giữ kết nối: `tools/colab_keepalive.sh omni`, chạy bằng `setsid`, ghi log ở `logs/keepalive.log`.
- Kiểm tra còn chạy: `ps -eo pid,args | grep "[t]ools/colab_keepalive"`, và `tail logs/keepalive.log` (mỗi 3 phút một dòng).
- Nếu nó đã chết thì khởi động lại ngay:
  `setsid nohup tools/colab_keepalive.sh omni > /dev/null 2>&1 < /dev/null &`
  Nếu script báo "already running" mà không có tiến trình nào, thì là do một tiến trình `sleep 180` mồ côi còn giữ khoá. Giết nó trước.
- **Khi trích xong:** chép đặc trưng về, rồi `touch logs/keepalive.stop` và `~/.local/bin/colab stop -s omni` để thôi tốn unit.

Lấy kết quả về máy, chọn một trong hai cách:
- `colab download`, mỗi lần một file. File nhỏ, khoảng 1 MB, nên tải từng file được.
- Nén thành vài file tar trên Colab, đẩy lên repo HF `nguyenminh04/uniav-youcook2-data` (token ở dòng 3 của `.env`), rồi kéo về.

---

## 2. Việc tiếp theo

1. **Đưa đặc trưng InternVideo2 vào dataset.**
   - `libs/datasets/youcook2_cap.py` hiện chỉ đọc file `_one_peace_*.npy`, lưới 0.5 giây, `feat_stride 8`, `num_frames 16`, 16 fps.
   - Đặc trưng InternVideo2 là 1 vector mỗi giây. Cần một tuỳ chọn nguồn đặc trưng trong config, đổi lưới thời gian tương ứng, và co giãn từng loại về 256 bước như hiện tại.
2. **Train và so ba cấu hình,** dùng `--set` và hồ sơ chạy:
   - chỉ InternVideo2,
   - InternVideo2 ghép ONE-PEACE (ghép theo kênh),
   - mốc hiện tại là `omni65`.
   - Nhớ chạy 2 seed cho cấu hình tốt nhất.
   Người dùng muốn **bỏ hẳn ONE-PEACE nếu InternVideo2 không kém hơn**.
3. **Cập nhật API:** thêm `uniav_api/encoders/internvideo2.py` theo interface `encoders/base.py`. Logic trích đã có sẵn trong `tools/extract_internvideo2.py`.
4. Các hướng khác đã bàn:
   - dùng InternVideo3-8B làm bộ sinh caption cho từng đoạn, thay cho việc chọn câu trong kho,
   - fine-tune thầy OmniRetriever bằng LoRA. Nên thử trước một lớp chuyển đổi nhỏ (MLP) trên vector có sẵn.

---

## 3. Kết quả hiện tại (model sự kiện, tập val YouCook2 gồm 394 video)

| Lượt | Thầy | R@0.5 | ret_sim | ret_sim câu đứng đầu | CIDEr | METEOR |
|---|---|---|---|---|---|---|
| `orig_baseline` (code gốc) | không | 49.6 | 0.717 (câu đứng đầu) | — | — | — |
| `sched10` | không | 49.3 | 0.748 | 0.722 | 88.0 | 14.4 |
| `omni65` (**checkpoint API đang dùng**) | có | 48.7 | **0.758** | **0.740** | 86.5 | **15.4** |
| `omni100`, `omni100_seed2`, `_w05`, `_avonly` | có | 47 đến 50 | 0.753 đến 0.755 | 0.735 đến 0.741 | 84 đến 86 | 15.1 |

Nhiễu giữa các seed: R@0.5 khoảng ±2, CIDEr khoảng ±5, ret_sim khoảng ±0.003.

**Kết luận:**
- **Phần tách đoạn không hơn code gốc.** R@0.5 vẫn quanh 49.
- **Phần mô tả tăng rõ:** ret_sim từ 0.717 lên 0.758, chủ yếu nhờ loss so với toàn kho, chọn câu theo đồng thuận, và thầy OmniRetriever.
- **Học trò đã đi được khoảng nửa quãng tới thầy.** Trên 500 clip GT tập val, thầy OmniRetriever-7B zero-shot đạt ret_sim 0.767, học trò 0.758, không thầy 0.748. Xem `experiments/teacher_vs_student.json` và `experiments/plots/`.
- **Nút thắt là đặc trưng đầu vào.** Nhánh video ONE-PEACE K400 không căn chỉnh với văn bản, và model overfit từ epoch 7. Đó là lý do chuyển sang InternVideo2.

---

## 4. Bản đồ code

| File | Vai trò |
|---|---|
| `train_event.py` | Train và chấm. Các cờ: `--set k=v` ghi đè config; `--eval ckpt` chấm một checkpoint; `--finalize` chấm lại lượt đã xong; `--note`. Mỗi lượt ghi hồ sơ `experiments/runs/<máy>-<lượt>.json` (commit, config, lịch sử, kết quả chấm cuối). Lịch mặc định 10 epoch. |
| `libs/modeling/event_archs.py` | Model: các đầu sự kiện, biên + IoU, nhúng, đoạn kèm bối cảnh video; tuỳ chọn thầy OmniRetriever (`omni_emb_file`) |
| `libs/datasets/youcook2_cap.py` | Dataset: kho caption, chỉ số caption trong kho, vector thầy |
| `configs/youcook2_event.yaml` | Cấu hình. Muốn bật thầy thì dùng `--set dataset.omni_emb_file=./data/youcookii/omni_emb_full.npz` |
| `tools/summarize_runs.py`, `tools/plot_runs.py` | Sinh bảng `RESULTS.md` và đồ thị |
| `tools/compare_teacher.py` | So học trò với thầy trên 500 clip GT |
| `tools/omni_youcook2.py` | Cắt clip, tạo manifest, đo thầy zero-shot |
| `tools/extract_internvideo2.py` | Trích InternVideo2 (không sửa repo upstream) |
| `tools/embed_captions.py` | Sinh `caption_emb.npz` |
| `tools/colab_keepalive.sh` | Giữ phiên Colab sống |
| `uniav_api/` | **API dạng hàm Python** (người dùng không muốn web API). Xem mục 5. |

---

## 5. API `uniav_api`

```python
import numpy as np, uniav_api as uv
r = uv.describe_video('data/demo_videos/6uHoTJSLoL8.mp4')   # chạy cả encoder; RTX 3060 mất ~1.8 lần độ dài video
r = uv.describe_features(v, a, duration, video_id='...')    # dùng đặc trưng có sẵn, khoảng 1 giây
uv.show(r)               # bảng: bước thật cạnh đoạn dự đoán, kèm thanh thời gian (người dùng đã duyệt định dạng này)
uv.compare_table(r)      # cùng bảng, dạng DataFrame
uv.search('boil the noodles', top_k=5); uv.embed_text('add salt')
```

- **Môi trường:** `/home/minh/uniav-api-env` (Python 3.11, torch 2.5.1 cu124). Kernel Jupyter tên `uniav-api-env (3.11)`.
- **Kiểm tra đã làm:**
  - đặc trưng audio trùng tuyệt đối với lúc train (cosine 1.00000), video gần tuyệt đối (0.99996),
  - soft-NMS viết bằng numpy giống bản C++ ở 200/200 trường hợp,
  - R@0.5 trên tập val = 48.70, đúng bằng lúc train,
  - chạy từ video gốc và từ đặc trưng có sẵn cho cùng 7/7 caption.
- **Ngưỡng chọn sự kiện:** `min_score 0.40`, chồng lấn tối đa 0.3, chọn trên tập val: F1 0.476 ở IoU 0.5.
- **Thiết bị:** CUDA, rồi MPS, rồi CPU. CPU chạy được nhưng quá chậm với encoder ONE-PEACE: người dùng đã dừng test CPU sau 17 phút cho một clip 30 giây. Trên CPU chỉ nên dùng `describe_features`.
- **Video demo:** `data/demo_videos/` có 3 video val (`-Ju39A-G0Dk`, `6uHoTJSLoL8`, `W2gnFLOi_AQ`).

---

## 6. Môi trường

| Môi trường | Dùng cho |
|---|---|
| `/home/minh/uniav-env` | Train và chấm. Python 3.8, torch 1.11 cu113, numpy 1.23.5 |
| `/home/minh/uniav-api-env` | API. Python 3.11, torch 2.5.1 |
| Colab qua `~/.local/bin/colab` | Tài khoản có 259 unit; A100 tốn 5.3 unit/giờ |

---

## 7. Thông tin đăng nhập và vận hành (KHÔNG in `.env` ra)

- **`.env`, mỗi dòng một token:**
  - dòng 1: fine-grained PAT GitHub, push bị lỗi 403,
  - dòng 2: classic PAT GitHub, dùng để push,
  - dòng 3: token ghi HuggingFace.
- **Push:**
  `GIT_ASKPASS=~/.config/uniav/git_askpass.sh GIT_TERMINAL_PROMPT=0 git -c credential.helper= push origin test`
- **Colab không pull được:** thường do các file hồ sơ chạy chưa được git theo dõi trùng tên. Cất chúng sang chỗ khác và `git checkout -- libs/utils/nms_1d_cpu.egg-info` rồi pull lại.
- **`pkill -f` hay khớp vào chính shell đang chạy nó.** Dùng mẫu kiểu `[x]yz`. Dấu `.` trong mẫu là ký tự đại diện, nên không để chuỗi khớp nằm sau trong cùng một lệnh.
- **Không lưu gì quan trọng trong scratchpad:** nó bị dọn khi Claude Code khởi động lại. Tiến trình nền phải chạy bằng `setsid`.
- **Upload lên Colab:** file lớn phải chia mẩu 64 MB, thư mục đích phải tồn tại trước. Tốc độ upload từ nhà khoảng 5 đến 9 MB/s.
- **HF dataset:** tài khoản miễn phí giới hạn 1000 yêu cầu mỗi 5 phút. Nên đóng nhiều file nhỏ thành tar trước khi đẩy lên.
- **METEOR trên Colab:** java hay chết giữa chừng. Đã xử lý bằng cách chạy METEOR trong tiến trình con có giới hạn thời gian; trên Colab cột METEOR bị bỏ trống.

---

## 8. Dữ liệu

| Ở đâu | Gì |
|---|---|
| `data/youcookii/av_features/` | Đặc trưng ONE-PEACE của 1500 video (11 GB); bản sao trên HF `nguyenminh04/uniav-youcook2-data` |
| `data/youcookii/caption_emb.npz` | Vector ONE-PEACE text của 11594 caption |
| `data/youcookii/omni_emb_full.npz` | Vector thầy OmniRetriever: 8218 text + 9060 av (train đủ, thiếu 3 clip hỏng; cộng 500 clip val) |
| `data/youcook2_cut/videos/` | Clip sự kiện đã cắt sẵn (17.7 GB) |
| HF `nguyenminh04/omni-model`, `nguyenminh04/omni-adapters` | Trọng số WAVE-7B (chưa vá `rope`, cần chạy `patch_wave_rope.py`) và adapter |
| Drive `MyDrive/code KL/Omni/data/YouCookII/videos` | 1500 video gốc chưa cắt (32.5 GB); gắn Drive cần người dùng tự chạy `colab drivemount` |
| `ONEPEACE_extract_embd_code/models/` | 3 checkpoint ONE-PEACE (video K400, audio, text) |
| `ckpt/` | `omni65`, `omni100*`, `sched10`, `base_seed2`, `orig_baseline`, cùng checkpoint UniAV gốc |

---

## 9. Thói quen và yêu cầu của người dùng

- Trao đổi bằng **tiếng Việt**. **Code, chú thích và log đều bằng tiếng Anh.**
- Log và kết quả in ra phải **dễ đọc**: dạng bảng, gom thành một bảng thay vì bắt đối chiếu qua lại.
- **Hỏi link hoặc đường dẫn dữ liệu, không tự đoán.** Có thì dùng HF hoặc Drive thay vì tải lên lại từ máy.
- **Không để Colab tắt.** Nếu đang chờ việc gì lâu, phải báo tiến độ định kỳ, kiểm tra có bị treo không và sửa ngay.
- **Mọi lượt chạy phải được ghi hồ sơ,** ghi lại cải tiến của từng biến thể để kết hợp ở lượt sau.
- Người dùng sẵn sàng tăng độ phức tạp của model nếu cải thiện được độ chính xác.
- Thư mục gốc gọn gàng: tài liệu để trong `docs/`, log trong `logs/`, kết quả trong `experiments/`.
