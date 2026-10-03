# Đường đi của dữ liệu: từ file trên đĩa tới `data_dict`

Tài liệu này mô tả phía **dataset**, tức toàn bộ những gì xảy ra bên trong
`ActivityNetDataset.__getitem__` ([libs/datasets/anet.py](../libs/datasets/anet.py)),
cho tới lúc trả về một `data_dict` cho model.

Mọi con số là số thật của một video YouCook2: `--bv0V6ZjWI`, dài 332.46 giây.

Ba file liên quan:

| File | Vai trò |
|---|---|
| `libs/datasets/anet.py` | luồng chính, `__getitem__` |
| `libs/datasets/data_utils.py` | `truncate_feats`, `label_points` |
| `libs/datasets/loc_generators.py` | `PointGenerator` |

---

## Ký hiệu

| Ký hiệu | Nghĩa | Giá trị |
|---|---|---|
| `T0` | số mốc trong file npy gốc | 663 |
| `T` | số mốc sau khi co, bằng `max_seq_len` | 256 |
| `C` | số kênh đặc trưng | 1536 |
| `N` | số đoạn gán nhãn của video | 10 |
| `F` | số tầng kim tự tháp | 6 |
| `P` | tổng số mốc cả kim tự tháp | 504 |
| `K` | số lớp | 59 |

`P = 256 + 128 + 64 + 32 + 16 + 8 = 504`

---

## Tổng quan 6 bước

```
file .npy + .json
   |
   1. Đọc đặc trưng           -> (T0, C) cho mỗi luồng, cắt về cùng độ dài
   2. Tính lại tỉ giá thời gian -> feat_stride, num_frames, feat_offset
   3. Co giãn đặc trưng        -> (C, T)
   4. Đổi nhãn sang lưới       -> segments (N, 2) đơn vị ô lưới
   5. Cắt ngẫu nhiên (chỉ train) -> T giảm còn ~230..256, N có thể giảm
   6. Sinh mốc + gán nhãn      -> points, gt_cls_labels (P, K), gt_offsets (P, 2)
   |
data_dict
```

---

## Bước 1. Đọc đặc trưng

[anet.py:196-209](../libs/datasets/anet.py#L196-L209)

Mỗi video có hai file npy, đọc lên rồi cắt cả hai về độ dài ngắn hơn.

```
--bv0V6ZjWI_one_peace_video_finetune.npy   (663, 1536)
--bv0V6ZjWI_one_peace_audio.npy            (664, 1536)
                                 -> cắt    (663, 1536) cả hai
```

**Vì sao `T0 = 663`.** Đặc trưng được trích bằng cửa sổ trượt: fps 16, cửa sổ
rộng 16 khung, bước nhảy 8 khung.

```
332.46 giây × 16 fps = 5319 khung hình
(5319 − 16) // 8 + 1 = 663 cửa sổ
```

Minh hoạ với video tí hon 50 khung, cửa sổ 16, bước 8:

```
mốc 0: khung  0..15  |################
mốc 1: khung  8..23  |        ################
mốc 2: khung 16..31  |                ################
mốc 3: khung 24..39  |                        ################
mốc 4: khung 32..47  |                                ################
mốc 5: khung 40..55  -> chỉ còn 10 khung, không đủ 16 -> dừng
=> T0 = 5
```

Các cửa sổ **chồng lấn 50%** vì bước nhảy bằng nửa bề rộng.

---

## Bước 2. Tính lại tỉ giá thời gian

[anet.py:223-240](../libs/datasets/anet.py#L223-L240)

Vì bước 3 sắp co 663 mốc xuống 256, con số `feat_stride: 8` trong config không
còn đúng. Phải tính lại ba giá trị.

```python
feat_stride = ((T0 - 1) * 8 + 16) / 256     # = 20.75
num_frames  = feat_stride                   # = 20.75
feat_offset = 0.5 * num_frames / feat_stride # = 0.5
```

**Tử số đo tổng phạm vi khung hình mà đặc trưng gốc thực sự phủ**, không phải độ
dài video. Với ví dụ tí hon ở trên:

```
(5 − 1) × 8 + 16 = 48 khung  (khung 0..47)
video có 50 khung nhưng 2 khung cuối không cửa sổ nào với tới
```

Ép về 4 mốc thì `feat_stride = 48 / 4 = 12`, mỗi mốc mới gánh 12 khung và các
mốc **ghép sát nhau không chồng lấn**:

```
TRƯỚC (5 mốc, chồng lấn)      SAU (4 mốc, ghép sát)
mốc 0: khung  0..15            mốc 0: khung  0..11
mốc 1: khung  8..23            mốc 1: khung 12..23
mốc 2: khung 16..31            mốc 2: khung 24..35
mốc 3: khung 24..39            mốc 3: khung 36..47
mốc 4: khung 32..47
```

**Vì sao `num_frames = feat_stride`.** Nó là bề rộng lát mới. Công thức giải mã
ngược lấy mép trái rồi cộng nửa bề rộng để nhảy vào **giữa** lát. Dùng số 16 cũ
thì cộng thiếu, mọi biên đoạn lệch sớm 0.148 giây một cách có hệ thống.

**Tỉ giá khác nhau cho từng video**, vì mọi video đều bị ép vào đúng 256 mốc:

| Video | duration | `T0` | `feat_stride` | 1 mốc = ? giây |
|---|---|---|---|---|
| -eyDS81FADw | 146.9 | 292 | 9.16 | 0.572 |
| --bv0V6ZjWI | 332.5 | 663 | 20.75 | 1.297 |
| -lMphXmWvbk | 448.0 | 894 | 27.97 | 1.748 |

---

## Bước 3. Co giãn đặc trưng

[anet.py:243-262](../libs/datasets/anet.py#L243-L262)

```
np.load          (663, 1536)     T trước, C sau
.transpose()     (1536, 663)     C trước, T sau (quy ước PyTorch)
.unsqueeze(0)    (1, 1536, 663)  thêm chiều batch cho F.interpolate
F.interpolate    (1, 1536, 256)  nội suy tuyến tính dọc trục thời gian
.squeeze(0)      (1536, 256)     bỏ chiều batch
```

Kết quả: `feats = {'visual': (1536, 256), 'audio': (1536, 256)}`

Đánh đổi: video YouCook2 dài từ 44 tới 1106 giây, ép hết về 256 giúp ghép batch
dễ và chặn chi phí attention, nhưng độ phân giải thời gian thành **tương đối**.
Đo thực tế: trung vị 1.17 giây mỗi mốc, chỉ 14 trên 2990 đoạn ngắn hơn một mốc.

---

## Bước 4. Đổi nhãn sang lưới

[anet.py:277-281](../libs/datasets/anet.py#L277-L281)

```python
segments = video_item['segments'] * fps / feat_stride - feat_offset
```

Ba phần: nhân `fps` ra chỉ số khung, chia `feat_stride` ra chỉ số mốc, trừ
`feat_offset` để chuyển sang **hệ quy chiếu tâm**.

**Vì sao phải trừ `feat_offset`.** Bộ sinh mốc đánh số bằng số nguyên 0, 1, 2 và
mỗi số nằm ở **tâm** ô. Phép chia lại cho ra toạ độ tính từ **mép trái**.

Ví dụ với ô rộng 20 khung:

```
        ô 0                ô 1                ô 2
  |------------------|------------------|------------------|
khung 0        10       20        30       40        50
              tâm               tâm               tâm

Sự kiện thật: giây 1.875 -> 3.125, tức khung 30 -> 50
  khung 30 chính là TÂM ô 1, khung 50 chính là TÂM ô 2

  không trừ offset : lưới [1.5, 2.5]   không ứng với mốc nào
  có trừ offset    : lưới [1.0, 2.0]   đọc thẳng: từ mốc 1 tới mốc 2
```

Phép đổi **thuận nghịch chính xác** với công thức giải mã ở
[multimodal_archs_multi_task.py:627](../libs/modeling/multimodal_archs_multi_task.py#L627).
Giá trị âm là bình thường, đoạn bắt đầu từ giây 0 sẽ ra `−0.5`.

Kết quả:

```
segments  (10, 2)  float32   đơn vị ô lưới
labels    (10,)    int64     label_id trong 0..58
```

---

## Bước 5. Cắt ngẫu nhiên (chỉ lúc train)

[anet.py:325-329](../libs/datasets/anet.py#L325-L329) gọi
`truncate_feats` ở [data_utils.py:25](../libs/datasets/data_utils.py#L25)

Tăng cường dữ liệu. Chọn độ dài ngẫu nhiên trong `crop_ratio: [0.9, 1.0]` tức
230 tới 256, rồi cắt một cửa sổ ở vị trí ngẫu nhiên.

Đoạn bị cửa sổ chặt ngang: giữ lại nếu phần còn lại trên `trunc_thresh: 0.5`,
bỏ nếu mất quá nửa. Biên đoạn được cắt gọn rồi dịch về gốc toạ độ mới (trừ `st`).

Vòng lặp tối đa 200 lần để bảo đảm cửa sổ còn ít nhất một đoạn.

Gọi 5 lần trên cùng một video:

| Lần | `T` | Số đoạn | Biên đoạn đầu |
|---|---|---|---|
| 0 | 242 | 9 | [5.2, 23.7] |
| 1 | 234 | 9 | [-0.5, 17.7] |
| 2 | 231 | 10 | [3.7, 15.2] |
| 3 | 237 | 9 | [-0.5, 17.7] |
| 4 | 237 | 10 | [-0.5, 8.2] |
| eval | 256 | 10 | [5.7, 17.2] |

**Đây là lý do hai lần chạy train liên tiếp cho kết quả khác nhau.** Muốn so sánh
tất định phải chuyển dataset sang chế độ eval.

---

## Bước 6. Sinh mốc và gán nhãn

### 6a. `PointGenerator`

[anet.py:335](../libs/datasets/anet.py#L335), cài đặt ở
[loc_generators.py:27](../libs/datasets/loc_generators.py#L27)

Trả về danh sách `F = 6` tensor, mỗi cái `(T_i, 4)` với 4 cột
`(t, lo, hi, stride)`:

| Tầng | Số mốc | Vị trí `t` | Dài phụ trách | `stride` |
|---|---|---|---|---|
| 0 | 256 | 0, 1, 2, …, 255 | 0..4 | 1 |
| 1 | 128 | 0, 2, 4, …, 254 | 4..8 | 2 |
| 2 | 64 | 0, 4, 8, …, 252 | 8..16 | 4 |
| 3 | 32 | 0, 8, 16, …, 248 | 16..32 | 8 |
| 4 | 16 | 0, 16, 32, …, 240 | 32..64 | 16 |
| 5 | 8 | 0, 32, 64, …, 224 | 64..10000 | 32 |

Cột `t` dùng **lưới gốc 0..255** chứ không phải chỉ số cục bộ, nhờ vậy sáu tầng
cùng một hệ toạ độ. Dải `lo..hi` lấy từ `regression_range`, khai báo cứng ở
[libs/core/config.py:31](../libs/core/config.py#L31) chứ không có trong yaml.

Phân công: tầng mịn nhiều mốc nhưng chỉ lo đoạn ngắn, tầng thô ít mốc lo đoạn dài.

### 6b. `label_points`

[data_utils.py:126](../libs/datasets/data_utils.py#L126)

Nối 6 tầng thành `(P, 4) = (504, 4)` rồi quyết định mốc nào phụ trách đoạn nào.
Ba điều kiện:

1. **Nằm trong đoạn.** Khoảng cách tới cả hai biên phải dương.
2. **Đúng dải độ dài.** Khoảng cách lớn hơn phải nằm trong `lo..hi` của tầng đó.
3. **Chọn đoạn ngắn nhất.** Nếu một mốc thoả nhiều đoạn, lấy đoạn ngắn nhất vì
   dễ hồi quy nhất. (Chỉ khi `class_aware: False`, đúng trường hợp YouCook2.)

Kết quả cuối, offset được **chia cho `stride` của tầng** để chuẩn hoá:

```
gt_cls_labels   (504, 59)  float32   one-hot lớp cho từng mốc
gt_offsets      (504, 2)   float32   khoảng cách tới 2 biên, đã chia stride
```

Trên video mẫu (chế độ eval, không cắt ngẫu nhiên), chỉ **41 trên 504 mốc là
dương**, tức 8.1%. Tính theo ô thì 41 trên 29736 ô, tức **0.138%**. Đây là lý do
phải dùng focal loss.

Phân bố mốc dương theo tầng:

| Tầng | Số mốc | Số mốc dương |
|---|---|---|
| 0 | 256 | 1 |
| 1 | 128 | 17 |
| 2 | 64 | 21 |
| 3 | 32 | 2 |
| 4 | 16 | 0 |
| 5 | 8 | 0 |

Hai tầng thô nhất không nhận gì, vì YouCook2 không có đoạn nào đủ dài.

---

## `data_dict` cuối cùng

[anet.py:314-341](../libs/datasets/anet.py#L314-L341)

| Khoá | Shape | Ghi chú |
|---|---|---|
| `video_id` | str | id YouTube |
| `feats` | dict: `visual (C, T)`, `audio (C, T)` | (1536, 256) |
| `segments` | `(N, 2)` | đơn vị ô lưới, có thể âm |
| `labels` | `(N,)` | `label_id` trong 0..58 |
| `points` | list `F` tensor `(T_i, 4)` | tổng 504 mốc |
| `gt_cls_labels` | `(P, K)` = (504, 59) | one-hot |
| `gt_offsets` | `(P, 2)` = (504, 2) | đã chia stride |
| `fps` | int | 16 |
| `duration` | float | 332.46 |
| `feat_stride` | float | **20.75**, không phải 8 |
| `feat_num_frames` | float | 20.75 |

`trivial_batch_collator` **không gộp tensor**, nó chỉ gom các `data_dict` thành
một danh sách. Nên `video_list` mà model nhận là `list` độ dài `B`.

---

## Ba chỗ dễ nhầm

**`feat_stride` trong `data_dict` khác `feat_stride` trong config.** Config là 8,
giá trị thực tế là 20.75 và khác nhau theo từng video.

**`segments` không phải đơn vị giây.** Nó là ô lưới, và giá trị âm là hợp lệ.

**Chạy train hai lần ra kết quả khác nhau.** Do `truncate_feats` cắt ngẫu nhiên.
Dùng chế độ eval khi cần so sánh tất định.

---

## Nguồn gốc

Toàn bộ logic trong tài liệu này **thừa kế từ ActionFormer** (ECCV 2022), UniAV
chép gần như nguyên văn. Ý tưởng point-based gốc từ **FCOS** (ICCV 2019), vốn là
phát hiện vật thể 2D, ở đây bỏ đi một chiều không gian.

Muốn hiểu sâu phần này thì đọc ActionFormer, bài báo UniAV không giải thích
vì nó không phải đóng góp của họ.
