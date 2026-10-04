# UniAV mới: luồng hoạt động qua một ví dụ

Tài liệu đi kèm sơ đồ [UniAV_new.drawio.svg](UniAV_new.drawio.svg). Mô tả đi từ trên xuống theo khung
"New UniAV: overview". Kích thước và thời gian lấy đúng theo cấu hình hiện tại
(`configs/youcook2_event.yaml`, `feat_source: iv2`). Các điểm số trong ví dụ là số minh hoạ.

---

## Giả sử có một video nấu cơm chiên dài 180 giây

Trong video có các bước: cắt cà rốt và hành, đánh trứng, chiên trứng, cho cơm vào đảo, nêm nước tương,
bày ra đĩa. Model không biết trước các bước này. Nhiệm vụ của nó là tự tìm **các đoạn có sự kiện**, rồi
gán cho mỗi đoạn **một câu mô tả**.

### Bước 1: Trích đặc trưng (hai khối tím, đóng băng)

- **Hình:** giải mã ở 2 khung hình/giây, ra 360 khung hình 224×224.
  - InternVideo2-1B lần lượt nhìn từng cửa sổ 1 giây, mỗi cửa sổ gồm 4 khung hình.
  - Mỗi giây cho ra một vector 768 số, mô tả "trong giây này đang thấy gì": dao cắt cà rốt, chảo có trứng...
  - Kết quả là ma trận 180 × 768.
- **Tiếng:** tách audio ở 16 kHz.
  - BEATs nghe cửa sổ 3 giây quanh mỗi giây (tiếng dao thớt, tiếng xèo xèo của chảo...).
  - Kết quả cũng là 180 × 768.
- **Chuẩn hoá:** mỗi vector được chuẩn hoá L2, rồi co giãn từ 180 dòng thành **256 bước**. Mỗi bước bây giờ
  ứng với khoảng 0.7 giây. Video dài hay ngắn đều được đưa về 256 bước như vậy.

Code: `tools/extract_internvideo2.py` (trích), `libs/datasets/youcook2_cap.py` (chuẩn hoá, co giãn).

### Bước 2: Backbone, cho hình và tiếng "trao đổi" với nhau (giữ nguyên UniAV)

- Hai chuỗi 256 bước (hình và tiếng) đi qua vài lớp conv, rồi qua self-attention. Ở bước này mỗi bước nhìn
  được các bước khác trong cùng một chuỗi, nên biết được bối cảnh trước và sau nó.
- Sau đó là cross-attention giữa hai chuỗi.
  - Hình hỏi tiếng: lúc thấy chảo, có nghe tiếng xèo không?
  - Tiếng hỏi lại hình theo chiều ngược lại.
  - Nhờ đó, các bước "thấy chảo + nghe xèo" được biểu diễn chắc chắn hơn so với chỉ nhìn hình.
- Backbone còn **thu nhỏ dần** chuỗi thành 6 mức (gọi là kim tự tháp FPN):

| Mức | Số bước | Mỗi bước ≈ | Hợp với sự kiện |
|---|---|---|---|
| 0 | 256 | 0.7 s | rất ngắn |
| 1 | 128 | 1.4 s | ngắn |
| 2 | 64 | 2.8 s | ... |
| 3 | 32 | 5.6 s | ... |
| 4 | 16 | 11 s | ... |
| 5 | 8 | 22 s | rất dài |

- Ở mỗi mức, hình và tiếng được ghép lại thành vector 1024 số. Tổng cộng có 256 + 128 + ... + 8 =
  **504 "điểm quan sát"**.

### Bước 3: Ba head, mỗi điểm quan sát được hỏi ba câu

Lấy ví dụ điểm quan sát ở **giây thứ 95, mức 3**. Lúc này trong video đang đảo cơm.

| Head | Câu hỏi | Ví dụ trả lời |
|---|---|---|
| **Event Head** | ở đây có đang diễn ra một bước nấu không? | 0.80 |
| **Boundary Head** | bước đó bắt đầu cách đây bao xa, kết thúc sau bao lâu? Đoạn mình đoán khớp thật tới đâu? | lùi 11.8 s, tiến 9 s, tức đoạn **[83.2 s, 104 s]**; IoU quality 0.70 |
| **Embed Head** | ở đây đang có nội dung gì? (một vector 512 số) | vector gần nghĩa với "đảo cơm trong chảo" |

### Bước 4: Chấm điểm và lọc trùng (chỉ có lúc suy luận)

- Mỗi điểm quan sát đề xuất một đoạn, kèm điểm số = sự kiện^0.7 × IoU^0.3. Ví dụ ở trên:
  0.80^0.7 × 0.70^0.3 ≈ **0.77**.
  - Thành phần IoU giúp một đoạn "chắc là có sự kiện nhưng biên đoán ẩu" bị xếp thấp hơn đoạn có biên chuẩn.
- Có tới 504 đề xuất, nhiều đề xuất gần như trùng nhau, vì các điểm từ giây 90 đến 100 đều đoán ra khoảng
  [83, 104]. **Soft-NMS** giữ lại đề xuất tốt nhất và hạ điểm những đề xuất trùng nó.
- API lọc thêm: chỉ giữ đoạn có điểm từ 0.40 trở lên và chồng lấn không quá 30% (`uniav_api/config.py`).
- Kết quả, ví dụ: **7 đoạn**, trong đó có [83.2 s, 104 s].

### Bước 5: SegmentContext, biến mỗi đoạn thành một vector q

Lấy đoạn [83.2 s, 104 s], tương ứng các bước 118 đến 148 trên lưới 256 bước.

- **Trung bình đoạn:** lấy trung bình các vector Embed Head (mức 0) từ bước 118 đến 148. Câu hỏi nó trả lời:
  "trong 20 giây này chủ yếu xảy ra gì?" Đáp án: đảo cơm.
- **Trung bình cả video:** "video này nấu món gì?" Đáp án: cơm chiên. Thông tin này giúp phân biệt "đảo cơm"
  với "đảo mì", hay "cho nước tương" với "cho nước sốt cà chua".
- **Vị trí:** đoạn nằm ở 46% đến 58% video, tức khoảng giữa. Bước giữa thường là nấu chính, không phải sơ chế
  hay bày đĩa.
- Ba thông tin này qua một MLP nhỏ, rồi **cộng vào trung bình đoạn**: MLP chỉ "chỉnh" thêm, không thay hẳn.
  Lớp cuối của MLP khởi tạo bằng 0, nên lúc mới train q chính là trung bình đoạn. Kết quả là vector **q**
  gồm 512 số.

### Bước 6: Chọn câu trong kho

- Kho có **8218 câu** lấy từ tập train, ví dụ "add the rice to the pan and stir", "cut the carrots into
  cubes"...
- Mỗi câu được chấm điểm bằng cosine giữa q và vector của câu, trong không gian caption của model (vector
  ONE-PEACE text qua `clip_proj`).
  - Có thể cộng thêm điểm trong không gian của thầy: q đi qua `omni_proj`, rồi so với vector text của thầy
    (đã tính sẵn cho 8218 câu). Đo trên tập val, cách này không giúp được gì đo được: ret_sim chênh không
    quá 0.003, nằm trong mức nhiễu. Vì vậy `uniav_api` chỉ dùng điểm trong không gian caption.
  - Tác dụng chính của thầy là lúc train: dạy q qua `omni_proj` (xem phần cuối).
- Lấy **20 câu điểm cao nhất**. Thay vì lấy câu đứng đầu, model chọn câu được **nhiều ứng viên khác đồng
  thuận** nhất (MBR).
  - Ví dụ: "add oil to the pan" có thể đứng đầu một cách ngẫu nhiên.
  - Nhưng nếu "add rice and stir fry", "stir the rice", "mix rice with vegetables" cùng nằm trong top và giống
    nhau, thì cụm này mới đáng tin, và model chọn câu trung tâm của cụm.

### Kết quả

```
[ 10.5 s –  34.8 s]  cut the carrots and onions into small pieces
[ 36.0 s –  48.2 s]  crack eggs into a bowl and whisk
[ 50.1 s –  68.7 s]  pour the eggs into the pan and scramble
[ 83.2 s – 104.0 s]  add the rice to the pan and stir fry
...
```

---

## Lúc train thì khác ở đâu

Video train có sẵn đáp án, ví dụ "giây 84–105: *add the rice and stir*". Luồng vẫn như trên, có thêm phần
chấm lỗi (các nhãn xanh trên sơ đồ):

1. **L_event:** các điểm nằm trong một bước GT phải trả lời "có sự kiện", các điểm ngoài phải trả lời
   "không". Mỗi mức FPN chỉ phụ trách sự kiện có độ dài hợp với nó, nên bước dài 21 giây do mức 3 hoặc 4 lo.
2. **L_reg:** khoảng cách tới đầu và cuối phải khớp với 84 và 105.
3. **L_iou:** điểm IoU quality phải bằng mức khớp thật giữa đoạn model đoán và đoạn GT. Nhờ vậy lúc suy luận
   nó mới đáng tin để xếp hạng.
4. **L_emb:** vector Embed Head ở mỗi điểm trong đoạn phải gần câu "add the rice and stir" hơn 8217 câu còn
   lại.
   - Các câu cùng nghĩa như "stir fry the rice" chỉ bị phạt nhẹ, nhờ nhãn mềm.
5. **L_span:** khác với lúc suy luận, SegmentContext nhận **đoạn GT** [84, 105], cộng thêm một **bản lệch
   biên** như [80, 109].
   - Vector q của cả hai đều phải chọn đúng câu.
   - Bản lệch tập cho model quen với việc lúc suy luận biên đoán ra sẽ hơi sai.
6. **Thầy OmniRetriever-7B:** từ trước khi train, clip [84, 105] đã được cắt ra và cho thầy xem, cả hình lẫn
   tiếng (`tools/omni_youcook2.py`).
   - **L_omni_av:** q (sau khi chiếu sang không gian thầy) phải gần cách thầy hiểu đúng clip này hơn tất cả
     clip khác. Đây là chỗ thầy "dạy cách nhìn".
   - **L_omni_txt:** q cũng phải chọn đúng câu theo cách thầy hiểu câu chữ.

Sau khi train xong, thầy không cần nữa: điều nó dạy đã nằm trong trọng số của học trò.

Code của model và các loss: `libs/modeling/event_archs.py`. Trọng số các loss: L_event, L_reg, L_iou là 1.0;
L_emb, L_span, L_omni_txt, L_omni_av là 0.2.
