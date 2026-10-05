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
| COIN | đang tải về PC (`datasets/annotations/download_videos.py`, yt-dlp trong `uniav-api-env`, ≤480p) | 11,827 video, 468 giờ, 46,354 đoạn, 749 nhãn khác nhau (TB 4.9 từ, lặp ~60 lần), 180 việc |
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
