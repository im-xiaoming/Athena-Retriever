# Tách bước và gán caption cho video nấu ăn YouCook2 (bản rút gọn)

Bản này để đọc hiểu mô hình: tìm các bước trong video nấu ăn rồi chọn caption cho từng bước.
Chỉ giữ phần cốt lõi, cấu trúc thư mục và backbone giữ như UniAV. Code đầy đủ với mọi tùy chọn thử nghiệm
nằm ở các nhánh train.

## Cấu trúc

```
libs/core/config.py            toàn bộ tham số (đường dẫn, kích thước model, trọng số loss, train)
libs/datasets/
    datasets.py                đăng ký dataset, make_dataset, make_data_loader, make_generator (như UniAV)
    youcook2.py                dataset YouCook2: feature nội suy về 256 bước, bước GT, caption
    data_utils.py              cắt ngẫu nhiên lúc train, gán nhãn cho từng điểm của pyramid
    loc_generators.py          sinh các điểm của pyramid
libs/modeling/
    blocks.py                  các block của UniAV (MaskedConv1D, LayerNorm, attention, Transformer)
    multimodal_backbones.py    backbone UniAV: hai luồng video và audio, cross-attention, pyramid 6 tầng
    losses.py                  focal loss, DIoU loss
    event_archs.py             các head (event, boundary, embedding, ground), loss, suy luận
    models.py                  đăng ký backbone và model (như UniAV)
libs/utils/
    nms.py, csrc/nms_cpu.cpp   soft-NMS viết bằng C++
    lr_schedulers.py           warm-up rồi giảm theo cosine
    train_utils.py             optimizer, EMA, train một epoch, chấm điểm
train.py
```

## Dữ liệu

`data/youcookii/` đã có sẵn file nhãn và vector caption (`caption_emb_iv2j.npz`, vector text InternVideo2).
Feature nằm trên HF `nguyenminh04/uniav-youcook2-data` (`iv2_feats/iv2_feats_*.tar`): giải nén hết các tar vào
`data/youcookii/iv2_feats/`, mỗi video một file `<video_id>.npz` (video InternVideo2 `v768` + audio BEATs `a768`,
mỗi giây một hàng).

## Chạy

    pip install torch numpy pycocoevalcap
    cd libs/utils && python setup.py install --user && cd ../..
    python train.py

Checkpoint có R@0.5 tốt nhất được lưu ở `ckpt/minimal/model_best.pth.tar`.
