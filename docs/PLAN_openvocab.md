# Kế hoạch: model hiểu câu mới (open-vocabulary), áp dụng ý tưởng OV-AVEL, 2026-10-06

Bài báo tham khảo: `OV-AVEL/2411.11278v3.pdf` (Zhou và cộng sự, CVPR 2025), code trong `OV-AVEL/proposed_method/`.

## Quyết định của người dùng (2026-10-06)

| Câu hỏi | Chọn |
|---|---|
| Câu truy vấn khi chạy thật thuộc phạm vi nào | **mọi chủ đề**: phải hiểu được cả việc chưa từng có trong dữ liệu train |
| Câu có cần khớp với âm thanh không | **không**: âm thanh chỉ làm ngữ cảnh như hiện nay (`audio_6b.pth` chỉ có BEATs, không có lớp chiếu sang chữ) |
| Có dùng bộ OV-AVEBench (VGGSound) không | **không**: chỉ lấy ý tưởng và cách đánh giá |
| Train ở đâu | thử trên **Colab A100** sau khi sửa xong |

## Bài báo nói gì (những điểm áp dụng được)

1. Giữ nguyên phía chữ. Train phía chữ, hoặc thêm khối trộn phức tạp, làm tăng điểm trên nhãn đã thấy nhưng tụt mạnh trên nhãn mới (Bảng 2 và 5: lớp linear đạt 65.8 trên nhãn đã thấy nhưng chỉ 28.3 trên nhãn mới).
2. Lớp temporal chỉ xử lý trong từng luồng tốt hơn trộn hai luồng (Bảng A9: 57.8 nếu chỉ trong luồng, 46.5 nếu chỉ trộn, 55.0 nếu cả hai). Một lớp tốt hơn nhiều lớp (Bảng A7).
3. Thêm một vector chữ `other` cho nền giúp tăng 10.8 điểm trung bình (Bảng 3).
4. Đánh giá tách nhãn **đã thấy** và **chưa thấy**: 46 lớp dùng khi train, 21 lớp chỉ có lúc test. Chỉ số: Acc theo từng giây, F1 theo đoạn, F1 theo sự kiện (IoU ≥ 0.5).
5. Dùng 25% dữ liệu train cũng cho kết quả gần bằng 100% (Bảng 6).

## Thay đổi trong model (đều là tùy chọn trong config, mặc định mới ghi rõ)

| Tùy chọn | Cũ | Mới | Lý do |
|---|---|---|---|
| `model.text_proj` | `linear` (train `clip_proj` biến đổi caption) | `none`: vector chữ InternVideo2 giữ nguyên, chỉ phía video học để khớp vào | điểm 1 |
| `model.pyramid_attn` | `cross` (các tầng pyramid chỉ trộn V với A) | thử `self` (mỗi luồng tự xử lý) | điểm 2, chạy so sánh |
| `train_cfg.loss_weight_other` | không có | các bước nền được kéo về vector `other` | điểm 3 |
| căn tâm vector chữ | trung bình caption YouCook2 | trung bình chung YouCook2 + COIN (`*_iv2j.npz`) | câu ngoài chủ đề nấu ăn không còn bị dồn về một hướng |

## Dữ liệu và cách đánh giá

- Train: YouCook2 train (1.106 video) cộng **COIN của các việc đã thấy**, phần `training`, trộn batch 1:1 theo nguồn. Mỗi nguồn có kho câu riêng trong loss (đúng `PLAN_multidataset.md`).
- Chia việc COIN: trong mỗi lĩnh vực (12 lĩnh vực, `taxonomy.xlsx`), khoảng 30% số việc thành **việc chưa thấy**. Video của các việc này không bao giờ dùng để train (tỉ lệ giống 21/67 của OV-AVEBench).
- Đánh giá:
  - YouCook2 val như cũ: R@0.5, CIDEr, METEOR, G R1@0.5.
  - COIN việc đã thấy (phần `testing`) và COIN việc chưa thấy (mọi video):
    - grounding: mỗi nhãn bước là một câu truy vấn, phải tìm đúng đoạn của nó (G R1@0.5, G R1@0.7);
    - nhận nhãn bước: vector của đoạn GT so với cả 743 nhãn cộng `other`, lấy top-1 (Acc);
    - F1 theo sự kiện: đoạn model tìm được và nhãn của nó phải khớp GT (IoU ≥ 0.5).
- Bỏ 3 video COIN trùng YouCook2 val (`data/coin_exclude.txt`).

## Bộ sinh câu (theo `PLAN_multidataset.md`, mục 4)

Đầu ra cuối cùng phải là caption do model sinh ra. Kho caption nấu ăn chỉ còn là tín hiệu phụ khi train, vì nó không mô tả được việc ngoài nấu ăn. Sau khi chọn được model sự kiện tốt nhất (các lượt R0 đến R4 bên dưới):

- Train lại GPT-2 (`tools/capgen/`) trên cả hai nguồn, có thẻ nguồn: `youcook2:` cho câu YouCook2, `coin:` cho nhãn COIN.
- Đo CIDEr và METEOR trên YouCook2 val. Đo trên COIN việc chưa thấy bằng cách so câu sinh ra với nhãn GT. Thử cả hai thẻ lúc suy luận.

## Thứ tự chạy trên Colab A100 (mỗi lượt 1 seed, sau đó lặp lại 2 seed cho lượt tốt nhất)

| Lượt | Nội dung |
|---|---|
| R0 | chỉ YouCook2, config hiện tại (có grounding head), làm mốc; đánh giá trên COIN như zero-shot |
| R1 | + COIN việc đã thấy |
| R2 | R1 + `text_proj: none` |
| R3 | R2 + `loss_weight_other` |
| R4 | R3 + `pyramid_attn: self` |
