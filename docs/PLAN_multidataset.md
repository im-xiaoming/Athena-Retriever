# Kế hoạch: thêm dataset (COIN, CrossTask) và sinh caption, 2026-10-05

Ghi lại các quyết định đã thống nhất với người dùng. Khi nào **xử lý xong dữ liệu**
(tải video, trích đặc trưng InternVideo2 + BEATs) thì mới làm tiếp các bước sau.

## Mục tiêu

- **Đầu ra là caption do model sinh ra**, không phải câu lấy lại từ kho câu train.
- Kho câu chỉ là công cụ phụ khi train: loss tương phản kéo vector đoạn về câu đúng để vector mang nghĩa.
- Tăng dữ liệu train bằng các bộ video hướng dẫn có mốc thời gian từng bước.

## Dữ liệu

| Dataset | Trạng thái | Ghi chú |
|---|---|---|
| COIN | **xong 2026-10-06**: 5.180 video có đặc trưng trên HF `nguyenminh04/coin-data` (`feats/`), 3 video trùng YouCook2 val ghi trong `data/coin_exclude.txt` | 11,827 video, 468 giờ, 46,354 đoạn, 749 nhãn khác nhau (TB 4.9 từ, lặp ~60 lần), 180 việc |
| CrossTask | chưa | video gỡ khỏi YouTube từ 2022, repo có link tải thay thế |
| HT-Step, VidChapters (phần nấu ăn) | để sau | chỉ làm nếu COIN giúp rõ |

Sau khi tải:
1. Trích đặc trưng bằng `tools/extract_internvideo2.py` (Colab A100, khoảng 16 giờ cho COIN).
2. **Loại video trùng YouTube ID với tập val của YouCook2** (nếu không, kết quả val sẽ ảo).

## Thiết kế đã chốt

1. **Một expert (một MLP) dùng chung cho mọi dataset.** Đây chính là backbone sau khi dọn trên nhánh `cleanup-iv2text`.
   - Không cần expert riêng theo dataset, vì các bộ dữ liệu cùng một bài toán (tìm bước trong video hướng dẫn) và cùng loại đặc trưng.
   - Expert riêng chỉ thử khi các cách dưới đây không đủ.
2. **Kho câu tách theo dataset trong loss:** đoạn COIN so với kho câu COIN, đoạn YouCook2 so với kho câu YouCook2.
3. **Cân bằng nguồn:** COIN có gấp khoảng 3 lần số đoạn so với YouCook2.
   - Trộn batch 1:1.
   - Nếu tách đoạn trên YouCook2 tụt: fine-tune thêm vài epoch chỉ trên YouCook2.
4. **Bộ sinh câu GPT-2 (`tools/capgen/`)** train trên cả hai nguồn, **có thẻ nguồn** trong đầu vào:
   - train: `[prefix] youcook2: caption:` → câu YouCook2; `[prefix] coin: caption:` → nhãn COIN;
   - suy luận: luôn dùng `youcook2:` để có câu chi tiết.

   Nếu không có thẻ, GPT-2 sẽ học viết câu cụt kiểu COIN. Ngoài CIDEr, phải xem câu bằng mắt để bắt lỗi bịa chi tiết (ví dụ "lamb" thay vì "pork").
5. **API:** chuyển mặc định sang `caption_mode='generate'`. Không có checkpoint bộ sinh câu thì quay về chọn câu từ kho kèm thông báo. Đã bàn nhưng chưa làm, chờ người dùng.
6. **Về sau:** train bộ sinh câu nối thẳng với model sự kiện, để loss sinh câu chảy vào backbone. Khi đó loss kho câu chỉ còn là tín hiệu phụ.

## Cách đánh giá

- Luôn chấm trên YouCook2 val (394 video), so với `iv2`, mỗi biến thể 2 seed.
- Nhiễu giữa các seed: ±1.5 R@0.5, ±0.003 txt_sim, ±2 CIDEr.
- Chỉ số chính cho caption: CIDEr và METEOR của câu sinh ra, trên các đoạn model tự tìm.

## Đã cân nhắc và bỏ

- **Ghép expert TAL + AVEL của UniAV gốc:** expert AVEL trong model của mình chưa bao giờ được train. Checkpoint gốc từng làm kết quả kém hơn hai lần (R@0.5 −3.7, CIDEr −17). Dữ liệu AVEL (UnAV-100) là sự kiện âm thanh chung, xa domain nấu ăn.
- **Đưa nhãn COIN vào kho câu khi suy luận:** model sẽ hay chọn câu ngắn, khái quát, làm CIDEr trên YouCook2 tụt.

## Đầu vào production (2026-10-05, nhánh `multimodal-query`)

Ba kiểu đầu vào: chỉ video, video + câu mô tả, chỉ câu mô tả (`uv.query`). Đã thêm **ground head** (câu điều khiển việc chấm điểm từng thời điểm, biên lấy từ boundary head, +2.9M tham số, `loss_weight_ground: 1.0`). Đã chạy thử pipeline trên PC, **chưa train**. Lần train tới cần xem thêm `G R1@0.5`, `G R1@0.7`, `G mIoU` bên cạnh R@0.5 và CIDEr, và kiểm tra ground head không làm tụt hai chỉ số kia. Với COIN: nhãn bước COIN cũng dùng làm câu truy vấn được.

## Nối với kế hoạch open-vocabulary (2026-10-06)

Người dùng chọn câu truy vấn thuộc **mọi chủ đề**. Kế hoạch chi tiết nằm ở `docs/PLAN_openvocab.md` (lấy ý tưởng từ bài OV-AVEL). Kế hoạch đó giữ nguyên mọi điểm của file này, cộng thêm:

- Chia việc COIN thành **đã thấy** và **chưa thấy**. Việc chưa thấy không dùng khi train và chỉ để đo khả năng hiểu câu mới.
- Phía chữ giữ nguyên, không train `clip_proj`. Căn tâm vector chữ bằng trung bình chung YouCook2 + COIN.
- Bộ sinh câu (mục 4 ở trên) càng cần thiết: chọn câu từ kho caption nấu ăn không mô tả được việc ngoài nấu ăn. Khi train bộ sinh câu, đo thêm trên COIN việc chưa thấy, và thử cả hai thẻ `youcook2:` và `coin:` lúc suy luận.
