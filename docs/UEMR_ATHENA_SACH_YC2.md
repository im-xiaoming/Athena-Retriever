# Athena sạch cho YouCook2 và chạy lại UEMR

Ngày lập: 2026-10-20. Lưu ở `Fushion/docs/UEMR_ATHENA_SACH_YC2.md`.

**Phạm vi:** chỉ YouCook2 (YC2), chỉ **UEMR**. Chưa đụng tới UEMR 2 (align) và HT-Step.

**Mục tiêu:** thay `athena.pth` hiện tại bằng một checkpoint train đúng quy trình, rồi chạy lại UEMR trên YC2 để kết quả chính của luận
văn trên YC2 là kết quả sạch.

File gồm hai phần:
- **Phần A:** train Athena sạch bằng code `UniAV-fixed`.
- **Phần B:** đưa checkpoint mới vào Fushion, rồi chạy lại UEMR.

---

## Quy tắc chung (áp cho cả phần A và B)

1. **Git:** được `git add` và `git commit` (commit nhỏ, message rõ việc gì). **Không `git push`**: người dùng tự push. Không `push --force`,
   không xoá nhánh, không rebase nhánh có sẵn.
2. **Chỗ chạy:**
   - **Local:** việc nhẹ, không cần GPU: đọc code, `torch.load(..., map_location="cpu")` để xem khoá checkpoint, so danh sách video, thống
     kê, chấm điểm và bootstrap trên vector đã cache, `pytest`.
   - **Colab:** mọi việc cần GPU (train Athena, encode Omni, train Context Fusion) và mọi việc cần dữ liệu chỉ có trên Drive. Agent viết cell,
     người dùng tự chạy và gửi lại log.
   - Trước mỗi việc, ghi vào nhật ký (mục C): local hay Colab, và vì sao.
3. **Cell Colab:** chạy lại được sau khi bị ngắt (bỏ qua mục đã có đầu ra hợp lệ), in môi trường ở đầu, mỗi cell kết thúc bằng `assert`.

---

## 0. Vì sao phải làm

`athena.pth` hiện tại (sha256 `2aef4406…`, biến thể `ov_E1`) có hai lỗi:

1. **Đã học 104 video dev.** Checkpoint được train trên 1.106 video train của YC2, gồm cả 104 video dev cùng câu của chúng.
2. **Được chọn checkpoint theo val**, tức chính tập 394 video dùng để báo cáo kết quả.

Với UEMR, hậu quả là: đoạn R6 do Athena cắt nên chất lượng cắt trên val có thể lạc quan; các tham số UEMR chọn trên dev, nơi Athena đã học,
nên cũng có thể nghiêng. Hội đồng chỉ cần hỏi "Athena có thấy dữ liệu đánh giá không?" là số R6 mất giá trị.

---

## Phần A — Train Athena sạch bằng `UniAV-fixed`

### A.0 Chỗ train và phạm vi sửa code

- **Code:** `D:\Học\KL\Code\UniAV\UniAV-fixed` (repo `im-xiaoming/Athena-Retriever`, nhánh `ov-refine`). Train trên Colab GPU.
- **Tạo nhánh mới** (ví dụ `yc2-clean`) từ đúng commit đã dùng để train `athena.pth`. Không làm trên `ov-refine` để không ảnh hưởng code của
  bạn phụ trách UniAV.
- **Chỉ được sửa:** danh sách video train, tập dùng để chọn checkpoint, và nơi tính `caption_pool.mean` (mục A.1–A.3). Sửa qua file config
  hoặc file danh sách video nếu có; chỉ sửa code khi không còn cách khác, và ghi rõ dòng nào.
- **Không sửa:** kiến trúc, loss, siêu tham số, feature đầu vào.
- **Việc đầu tiên** trước khi train: tìm trong code nơi định nghĩa (1) danh sách video train, (2) tập dùng để chọn checkpoint / dừng sớm,
  (3) nơi tính `caption_pool.mean`. Ghi ba vị trí (file + dòng) vào nhật ký (mục C) rồi mới sửa.

### A.0b Checkpoint pretrain `model_epoch_006.pth.tar` (local, không GPU)

Có một checkpoint pretrain `model_epoch_006.pth.tar` nhưng chưa biết nó khớp với Athena tới đâu và đã học dữ liệu gì. **Làm mục này trước
khi train.** Chỉ đọc, không sửa file.

**1. So khoá và kích thước** với cả hai bộ trọng số của `athena.pth` (`state_dict_seg` và `state_dict`):

```python
import torch
pre = torch.load("model_epoch_006.pth.tar", map_location="cpu", weights_only=False)
ath = torch.load("athena.pth", map_location="cpu", weights_only=False)
print("khoá cấp 1 pretrain:", list(pre.keys()) if isinstance(pre, dict) else type(pre))
sd_pre = pre.get("state_dict", pre.get("model", pre))      # tuỳ định dạng thực tế
sd_pre = {k.removeprefix("module."): v for k, v in sd_pre.items()}
for name in ("state_dict_seg", "state_dict"):
    sd = {k.removeprefix("module."): v for k, v in ath[name].items()}
    same  = [k for k in sd if k in sd_pre and sd[k].shape == sd_pre[k].shape]
    shape = [k for k in sd if k in sd_pre and sd[k].shape != sd_pre[k].shape]
    only_a = [k for k in sd if k not in sd_pre]
    only_p = [k for k in sd_pre if k not in sd]
    n = lambda ks, d: sum(d[k].numel() for k in ks)
    print(f"{name}: khớp {len(same)} khoá ({n(same, sd)/n(list(sd), sd):.1%} tham số) | "
          f"lệch shape {len(shape)} | chỉ Athena {len(only_a)} | chỉ pretrain {len(only_p)}")
```

In thêm danh sách khoá theo **tiền tố module** (ví dụ `backbone.*`, `neck.*`, `embed_head.*`, `omni_proj.*`) cho từng nhóm, và mọi
metadata trong file (epoch, config, tên dataset, args). Ghi kết quả vào nhật ký.

**2. Xác định nguồn gốc.** Trả lời ba câu, ghi vào nhật ký kèm bằng chứng (file config, commit, script train):
- `athena.pth` cũ có được khởi tạo từ `model_epoch_006.pth.tar` không? Tìm trong config/script train của `UniAV-fixed` (tham số kiểu
  `resume`, `pretrained`, `load_from`, `init_checkpoint`).
- Pretrain này học dữ liệu gì? Nếu là pretrain gốc của UniAV (nhiều bộ dữ liệu), xem bộ đó có gồm YouCook2 không.
- Nó có phải là một epoch sớm của chính Athena train trên YC2 (gồm cả dev) không?

**3. Quyết định khởi tạo cho Athena sạch:**

| Trường hợp | Làm gì |
|---|---|
| `athena.pth` cũ khởi tạo từ pretrain này **và** pretrain không học video dev/val YC2 | khởi tạo từ pretrain này, như cũ |
| Pretrain đã học video dev hoặc val YC2 (ví dụ là epoch sớm của Athena cũ) | **không dùng**; khởi tạo như cách đã dùng trước khi có pretrain này. Hỏi người dùng nếu không rõ |
| `athena.pth` cũ **không** khởi tạo từ pretrain này | giữ cách khởi tạo cũ; không đổi sang pretrain (A.4: giữ nguyên mọi thứ khác) |
| Không xác định được nguồn gốc | dừng, báo người dùng kèm kết quả bước 1 |

Khớp khoá và shape chỉ cho biết **có thể** nạp được, không cho biết **nên** nạp. Quyết định theo bảng trên, không theo tỉ lệ khớp.

### A.1 Dữ liệu

| Tập | Video | Dùng để |
|---|---|---|
| trainsplit | 1.002 (= 1.106 train − 104 dev) | train |
| dev | 104 (773 câu) | chọn checkpoint, chọn epoch |
| val | 394 (3.030 câu) | **không được dùng dưới bất kỳ hình thức nào** |

- Danh sách video từng tập lấy từ `omni_cache/common/subset_videos.json` (trường `split`). Dùng đúng danh sách này, không chia lại.
- **Không dùng val** để: train, chọn epoch, chọn checkpoint, dừng sớm, chọn siêu tham số, tính thống kê (mean, chuẩn hoá), hay xem số
  giữa chừng.
- **Nếu train multi-domain** (thêm dữ liệu ngoài YC2): kiểm tra dữ liệu thêm vào không chứa video nào thuộc dev hoặc val YC2 (so theo
  YouTube ID). Ghi danh sách nguồn dữ liệu đã dùng.

### A.2 Chọn checkpoint

- Chọn theo **dev**, dùng đúng tiêu chí đã dùng trước đây trên val (chỉ đổi tập).
- Athena cũ có **hai bộ trọng số**: `state_dict_seg` (epoch 7, dùng để cắt đoạn R6 và ngữ cảnh Z) và `state_dict` (epoch 10, dùng để lấy
  vector đoạn q). **Cả hai** phải chọn trên dev. UEMR chỉ dùng `state_dict_seg`, nhưng vẫn chọn và lưu cả `state_dict` để sau này làm UEMR 2
  không phải train lại.
- Ghi lại số trên dev của checkpoint được chọn. **Không** chạy checkpoint trên val trong phần A.

### A.3 Thống kê tính từ dữ liệu

- `caption_pool.mean` (trung bình vector câu IV2, dùng để trừ mean trong `athena/pipeline.py`): kiểm tra đang tính trên những câu nào. Nếu
  có câu val, **tính lại chỉ trên câu trainsplit**.
- Mọi thống kê khác (chuẩn hoá feature, bank caption, pool câu...) cũng chỉ lấy từ trainsplit.

### A.4 Giữ nguyên mọi thứ khác

Kiến trúc, siêu tham số, feature đầu vào (`nguyenminh04/uniav-youcook2-data`), seed: **giữ như lần train cũ**. Thay đổi duy nhất là tập train
và tập chọn checkpoint. Nếu buộc phải đổi gì, ghi ra kèm lý do.

### A.5 Bàn giao

| Thứ | Ghi chú |
|---|---|
| File checkpoint mới | cùng định dạng `athena.pth` (có `state_dict_seg`, `state_dict`) để `uniav_app` đọc được không cần sửa |
| sha256 | để đặt tên thư mục cache và kết quả |
| Commit code + file config đã dùng | nhánh `yc2-clean` của `UniAV-fixed`; kèm diff so với commit train `athena.pth` cũ |
| Danh sách video train thực tế | xác nhận đúng 1.002 video, không có dev, không có val |
| Epoch được chọn + số trên dev | cho từng bộ trọng số |

---

## Phần B — Chạy lại UEMR với Athena sạch

### B.1 Quy tắc

1. **Không ghi đè kết quả cũ:** `results/056d388e/` và mọi cache, kết quả của Athena cũ. Kết quả mới vào thư mục theo hash mới.
2. **Phần B không sửa gì trong `UniAV-fixed`.** Mọi thay đổi ở đó chỉ thuộc phần A, trên nhánh `yc2-clean`.
3. **Giữ nguyên cấu hình UEMR** trong `configs/final_protocol.json`. Lý do: tránh chọn lại sau khi đã thấy số, và để cùng một "công thức"
   UEMR sau này áp nguyên sang HT-Step.
4. **Ghi quyết định trước khi chấm val** (mục B.3) vào nhật ký (mục C).
5. Khi chưa chắc, hỏi người dùng; không thêm thí nghiệm ngoài file này. **Không chạy gì của UEMR 2.**

### B.2 Các bước

| # | Việc | Chỗ chạy | Ghi chú |
|---|---|---|---|
| 1 | **Đưa checkpoint vào Fushion:** chép lên Drive (hoặc HF) ở đường dẫn mới, cho `uniav_app` (`Config`, `download_checkpoint`) trỏ tới file mới qua tham số, **không ghi đè** `athena.pth` cũ. Kiểm sha256, đủ hai bộ trọng số, đọc được bằng `uniav_app` | local + Drive | so danh sách video train với `subset_videos.json` |
| 2 | Chạy Athena mới trên val + dev: đoạn R6 mới, ngữ cảnh Z mới | Colab | các cách cắt dựa trên UniAV (R6, `uni_M_gt`, `uniav_shift`) đều phải tạo lại; R0, R2, R3, R7 không đổi |
| 3 | **Encode Omni `ft` cho đoạn R6 mới** (val + dev) | Colab GPU | đoạn mới có biên khác nên không dùng lại được cache `e` cũ; chỉ encode đoạn chưa có trong cache (khoá theo `seg_key`) |
| 4 | Chạy lại `notebooks/uemr_final_AZ.ipynb` với Athena mới | Colab | xem B.4 |
| 5 | Viết kết quả (mục B.5) | local | |

### B.3 Quyết định phải ghi trước khi chấm val

| Câu hỏi | Đề xuất | Lý do |
|---|---|---|
| Cấu hình UEMR (`final_protocol.json`) có chọn lại trên dev sạch không? | **Không**, giữ nguyên | tránh chọn lại sau khi thấy số; giữ một công thức cố định cho cả YC2 và HT-Step |
| Context Fusion (dòng R6 + Z) | train lại với Z mới, **giữ nguyên cấu hình D3** đã chốt; dừng sớm theo dev như cũ | Z lấy từ Athena nên Z cũ không còn dùng được; dev giờ sạch nên dừng sớm theo dev là hợp lệ |
| Thứ tự chấm | chấm dev trước, kiểm tra hợp lý, rồi mới chấm val **một lần** | |

### B.4 Chạy lại `uemr_final_AZ.ipynb`

- Đổi checkpoint Athena sang file mới và đường dẫn đầu ra sang thư mục theo hash mới.
- **Kiểm tra bắt buộc:** các cột không phụ thuộc Athena phải trùng số cũ trên kho val 394: R0 8.32, R3 18.02, R2 27.16, R7 37.52. Nếu
  lệch, dừng lại tìm nguyên nhân trước khi đọc R6.
- Các dòng phụ thuộc Athena sẽ đổi: R6, R6 ∪ R2, R6 + Context Fusion, `uni_M_gt`, `uniav_shift`, VCMR R@1 của R6.
- Bỏ các assert tái lập số R6 cũ (24.88) nếu có.

### B.5 Kết quả cần giao

1. Bảng so sánh **cũ và mới** trên kho val 394 (và một dòng kho 498):

   | Dòng | Athena cũ (`2aef4406`) | Athena sạch (`<hash8>`) |
   |---|---|---|
   | R0 / R3 / R2 / R7 R@1 | 8.32 / 18.02 / 27.16 / 37.52 | phải trùng |
   | R6 R@1 / R@5 / R@10 | 24.88 / 52.05 / 64.72 | |
   | R6 VCMR R@1 | 16.93 | |
   | R6 ∪ R2 R@1 | 29.77 | |
   | R6 + Context Fusion R@1 | 25.31 | |
   | R6 − R3 (CI 95%) | | |

2. Cập nhật `UEMR_KET_QUA.md`: thay số YC2 bằng số sạch; giữ số cũ trong một mục "kết quả với Athena có rò rỉ" để đối chiếu.
3. Ghi rõ những gì **vẫn** là hạn chế: một bộ dữ liệu (HT-Step làm sau); Omni `ft` đã học GT trainsplit.

### B.6 Đọc kết quả

- R6 mới **thấp hơn** 24.88 là điều dễ xảy ra (rò rỉ cũ làm số đẹp lên); đó không phải lỗi.
- Điều cần giữ được cho luận văn là **thứ tự**: R6 (đoạn do UniAV cắt) vẫn tốt hơn các cách cắt không dùng UniAV có chi phí tương đương
  (R3, R0). So sánh với R2 và R7 như trong `UEMR_KET_QUA.md`.
- Nếu thứ tự đổi, dừng lại báo người dùng trước khi viết kết luận.

---

## Sau này (chưa làm trong file này)

- **UEMR 2 (align):** dùng `state_dict` của checkpoint sạch, train lại `omni_proj`, chạy lại `uemr2_align_AZ.ipynb`. Sẽ có hướng dẫn riêng.
- **HT-Step:** train UniAV trên HT-Step theo cùng quy tắc ở phần A, chạy UEMR với cấu hình giữ nguyên.

---

## C. Nhật ký

(Agent ghi vào đây: chỗ chạy của từng việc, vị trí code ở A.0, kết quả và quyết định ở A.0b, quyết định B.3, số kiểm tra B.4, kết quả B.5,
hash các commit đã tạo.)

### C.1 Chỗ chạy (2026-10-20)

| Việc | Chỗ | Vì sao |
|---|---|---|
| A.0 tìm vị trí code | local | chỉ đọc code, `git show 6552590:…` |
| A.0b so khoá `model_epoch_006.pth.tar` (2,1 GB) với `athena.pth` | local, CPU | `torch.load(map_location="cpu")`, máy 15 GB RAM; quy tắc 2 cho phép |
| So danh sách dev (`dev_videos.txt`, `youcookii_annotations_uemr.json`) | local | đọc json |
| Commit lên nhánh `chien` của `UniAV-fixed` | local, git | không push (người dùng tự push) |
| So với `subset_videos.json`, tính lại mean câu IV2 (encoder chữ IV2 2,8 GB), train, export | **Colab GPU** | cần GPU; `subset_videos.json` và manifest chỉ có trên Drive; feature ở HF |
| Phần B | Colab, sau khi có checkpoint | |

### C.2 Vị trí code (A.0), tại commit `6552590` (`train_event.py`; ở HEAD `48918c5` là `train.py`, cùng số dòng)

1. **Danh sách video train:** `configs/youcook2_event.yaml:5-6` (`train_split: [training]`) và `:10` (`dataset.json_file` =
   `youcookii_annotations_trainval.json`) → `train_event.py:405` → `libs/datasets/youcook2_cap.py:147` (`_load_json_db`: giữ video có
   `subset` thuộc split, có feature và đủ khoá câu). Dev nằm trong `training` nên được train.
2. **Tập chọn checkpoint:** `configs/youcook2_event.yaml:7-8` (`val_split: [validation]`) → `train_event.py:406, 424` (`dv`, `dlv`) →
   `train_event.py:584-605`: mỗi epoch `evaluate(net, dlv)`, lưu `best_seg` khi R@0.5 cao hơn, `best_cap` khi CIDEr cao hơn (không
   dừng sớm, đủ 12 epoch); `:619-622` `final_eval` của `best_cap` trên `dlv`. `tools/export_api_ckpt.py:66-75`: `state_dict` = `best_cap`,
   `state_dict_seg` = `best_seg` (chỉ khi khác epoch).
3. **`caption_pool.mean`:** `tools/embed_captions.py:77` (`joint()`: `mean = 0.5·mean(câu train duy nhất) + 0.5·mean(nhãn COIN train)`),
   tập câu lấy ở `:103` (`subset == 'training'`, video có feature) → **có 773 câu dev, không có câu val**. Lưu ở
   `data/youcookii/caption_emb_iv2j.npz['mean']`, export chép vào `ck['caption_pool']['mean']` (`export_api_ckpt.py:45`), dùng ở
   `athena/pipeline.py:154` cho câu mới. Đã kiểm: mean trong file git == `athena.pth` `caption_pool.mean` (Colab ô 5 kiểm lại).
4. Thống kê khác: pool câu (`youcook2_cap.py:96`, chỉ split train) và vector teacher Omni (`:68`, chỉ split train) tự theo split train.
   Feature chỉ chuẩn hoá L2 từng hàng (`:183`), không có mean/std của tập dữ liệu.

### C.3 A.0b: `model_epoch_006.pth.tar`

**Bước 1 (khoá, kích thước).** Khoá cấp 1: `epoch` (= 6), `state_dict` (697 tensor, 130,8M), `state_dict_ema`, `optimizer`, `scheduler`
(warmup 1.250 bước, tổng 4.375 bước, lr 1e-3). **Không có config, tên dataset hay args.** Module cấp 1: `backbone`, `cls_head`, `reg_head`.
`athena.pth`: `state_dict` (epoch 10) và `state_dict_seg` (epoch 7), mỗi bộ 526 tensor, 66,3M; run `ov_E1`, commit `6552590`.

| | `state_dict_seg` | `state_dict` |
|---|---|---|
| khớp tên + shape | 378 khoá (28,0% tham số): `backbone.cross_att_*` 230, `self_att_*` 92, `ori_cross_att_*` 46, `embd_norm_*` 8, `embd_*` 2 | như bên trái |
| lệch shape | 2: `backbone.embd_V/A.0.conv.weight` (512, **768**, 3) so với (512, **1536**, 3) | như bên trái |
| chỉ Athena | 146: `event_head`, `bound_head`, `embed_head`, `ground_head`, `seg_ctx`, phần còn lại của backbone | như bên trái |
| chỉ pretrain | 317: `cls_head.*`, `reg_head.*`, và trong mỗi khối backbone 3 nhánh `ln2_TAL / mlp_TAL`, `_AVEL`, `_SED` | như bên trái |
| cos(Athena, pretrain) trên 314 ma trận khớp shape | trung vị 0,007; chỉ các gain LayerNorm (khởi tạo = 1) có cos ≈ 0,997 | như bên trái |

**Bước 2 (nguồn gốc).**
- `athena.pth` **không** khởi tạo từ file này. Bằng chứng: `train_event.py` ở `6552590` không có tham số nạp trọng số nào (chỉ `--eval`,
  `--finalize`); `--pretrain` đã bị xoá ở commit `ecbc84f` ("drop the multi-task UniAV leftovers", tổ tiên của `6552590`); lệnh E1 trong
  `tools/colab_ov.sh` chỉ có `--set train_cfg.ema_decay=0.999`; trọng số gần như trực giao (cos trung vị 0,007); `experiments/NOTES.md`:
  "Original UniAV checkpoint (`--pretrain`): worse twice … no".
- File này là **checkpoint UniAV đa nhiệm gốc** (3 nhánh TAL / AVEL / SED, đầu `cls_head` / `reg_head`, feature vào 1.536 chiều). Theo bài
  báo UniAV, các nhiệm vụ đó dùng ActivityNet 1.3, UnAV-100, DESED, không có YouCook2; file không có metadata dataset nên **không kiểm
  được trực tiếp**. Vì không dùng nên điều này không ảnh hưởng.
- Không phải epoch sớm của Athena trên YC2: kiến trúc khác (3 nhánh chuyên gia đã bị bỏ 2026-10-05, 135M → 68M), đầu ra khác, chiều vào khác.

**Bước 3 (quyết định):** dòng 3 của bảng → **giữ cách khởi tạo cũ** (ngẫu nhiên, `init_rand_seed 1234567891`), không nạp pretrain.

### C.4 Phần A: code trên nhánh `chien` của `UniAV-fixed` (local, người dùng tự push)

**Đổi so với đề bài (theo yêu cầu người dùng 2026-10-20):** không dùng nhánh riêng `yc2-clean`; mọi thứ (code, notebook, file này) nằm
trong `UniAV-fixed`, nhánh `chien`, để người dùng push một lần. `chien` đang ở `48918c5`; so với commit train `athena.pth` (`6552590`)
chỉ khác **tên** (`train_event.py` → `train.py`, `uniav_api` → `athena`, chú thích), đã kiểm bằng `git diff 6552590 48918c5 -- configs
libs train_event.py train.py tools/embed_captions.py tools/export_api_ckpt.py`: không đổi dòng code nào của train, dataset, loss, export.

Commit trên `chien` (xem `git log`): "yc2-clean: train on trainsplit…" (code + dữ liệu) và "notebooks, docs: Athena sạch YC2 phần A".

| File | Thay đổi |
|---|---|
| `data/youcookii/annotations/youcookii_annotations_yc2clean.json` (mới) | = `Data/YouCookII/metadata/youcookii_annotations_uemr.json` (sha256 `0483e7bb…`): trainval json, chỉ 104 video dev đổi `subset` thành `"dev"` (đã kiểm local: cùng khoá, 0 video khác ngoài `subset`, tập `dev` == `dev_videos.txt`) |
| `configs/youcook2_event_yc2clean.yaml` (mới) | = `youcook2_event.yaml`, đổi `json_file` (file trên), `val_split: [dev]`, `caption_emb_file: caption_emb_iv2j_yc2clean.npz`, `coin.eval: false` |
| `tools/embed_captions.py` | **sửa code** (không có cách khác vì đường dẫn annotation và file ra bị viết cứng): `joint(anno, tag)` dòng 58, 71, 81, 84; `youcook2_captions(anno)` dòng 92, 95; `main` dòng 114. Lệnh cũ không đổi hành vi |
| `notebooks/athena_sach_yc2_A.ipynb` (mới) | notebook Colab phần A (thêm bằng `git add -f` vì `.gitignore` bỏ qua `*.ipynb`) |
| `docs/UEMR_ATHENA_SACH_YC2.md` (mới) | file này |

Thay đổi khác so với `ov_E1` và lý do: `coin.eval: false` (COIN test chỉ in thêm số, không chọn gì; tắt để khỏi tải COIN và khỏi lệch mean
với file nhãn COIN). Caption bank của API (`athena/assets/caption_pool.npz`, 8.218 câu train cũ, gồm text câu dev) **giữ nguyên** vì
`uniav_app` kiểm số câu bằng file assets (`captioner.py:45`); vector được căn theo mean sạch. Bank chỉ dùng khi in caption; UEMR không dùng
(R6 và Z chỉ dùng `state_dict_seg`). Lệnh train: `train.py configs/youcook2_event_yc2clean.yaml --output yc2clean_E1 --set
num_workers=4 train_cfg.ema_decay=0.999` (đúng lệnh E1).

Notebook Colab: `notebooks/athena_sach_yc2_A.ipynb` (ô 1–8), clone nhánh `chien` mới nhất. Ô 4 kiểm train == 1.002 video của manifest
trainsplit, dev == 104 video dev của `subset_videos.json`, train ∩ (dev ∪ val) = ∅, bằng đúng lớp dataset `train.py` dùng. Ô 5 trước hết
encode lại file câu cũ để chắc encode tái lập được (cos > 0,999), rồi mới tính file sạch. Ô 6 kiểm log in `1002 train / 104 val videos`,
config không nhắc `validation`, code không dirty (commit lúc train ghi trong `record.json`). Ô 7 export, so định dạng với `athena.pth`
(cùng khoá, shape, config), ghi sha256, `<hash8>/athena_yc2clean.pth` và `SOURCE.json` (epoch + số dev của cả hai bộ trọng số) vào
`MyDrive/…/Fushion/uemr/athena_sach/`. **Val không được chấm ở đâu trong phần A.**

**Đang chờ:** người dùng push nhánh `chien`, chạy notebook, gửi log. Sau đó ghi ở đây: sha256, epoch được chọn và số dev (A.5).

### C.5 Quyết định B.3 (ghi trước khi chấm val; chưa chấm val với Athena mới)

| Câu hỏi | Quyết định |
|---|---|
| Chọn lại `final_protocol.json` trên dev sạch? | **Không**, giữ nguyên mọi khoá |
| Context Fusion | train lại với Z mới, giữ cấu hình D3, dừng sớm theo dev như cũ |
| Thứ tự chấm | dev trước, kiểm hợp lý, rồi val **một lần** |
| θ hậu xử lý UniAV | **đề xuất:** áp lại quy tắc có sẵn trong protocol (`uniav.postprocess.theta`: trên dev, mean K_pred gần mean K_GT nhất, hoà → θ nhỏ hơn); 0,40 là giá trị của checkpoint cũ (`value_current_uniav`). Quy tắc chỉ dùng số đoạn trên dev, không nhìn val. **Chờ người dùng xác nhận** |
| Đường dẫn | checkpoint ở `uemr/athena_sach/<hash8>/`; đầu ra UniAV ở `uemr/uniav_delivery_<hash8>/`; A-Z (`UNIAV_SOURCE = 'new'`) ghi theo hash đầu ra, không đụng `results/056d388e/` |
