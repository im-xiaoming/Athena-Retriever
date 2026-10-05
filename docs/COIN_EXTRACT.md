# Trích đặc trưng COIN trên nhiều GPU cùng lúc, 2026-10-05

Script điều phối là `tools/coin_hub.py` (nhánh `coin-extract`). Repo HF trung chuyển:
`nguyenminh04/coin-data`, private, dạng dataset.

| Thư mục trên HF | Ai ghi | Nội dung |
|---|---|---|
| `videos/coin_videos_NNNN.tar` | PC (`coin_hub.py upload`, chạy liên tục khi đang tải) | 100 video mp4 ≤480p mỗi gói, khoảng 0.8 GB |
| `videos/manifest.json`, `videos/DONE` | PC | gói → danh sách video; DONE chỉ được ghi khi chạy tay `coin_hub.py upload --final` sau khi đã tải xong |
| `claims/NNNN__<tên máy>` | máy GPU | máy nào đang làm gói nào (claim). Máy claim trước thắng; claim cũ hơn 6 giờ coi như đã chết |
| `feats/coin_feats_NNNN.tar` | máy GPU | mỗi video một `<id>.npz` (v768, v512, a768; float16; 1 dòng/giây), kèm `failed.txt` |

Mỗi máy GPU tự lấy gói chưa ai làm, nên **thêm hay bớt máy lúc nào cũng được**. Máy tự thoát khi PC đã ghi DONE và mọi gói đều có `feats`.
Tốc độ đo được: 5.9 giây video mỗi giây trên RTX 3060. COIN có khoảng 468 giờ video, nên một GPU cỡ T4 cần khoảng 80 giờ; 3 GPU thì khoảng 27 giờ.

## Dựng một máy GPU (Colab T4 hoặc Kaggle)

```bash
git clone -b coin-extract https://github.com/im-xiaoming/UniAV-fixed.git UniAV && cd UniAV
git clone --depth 1 https://github.com/OpenGVLab/InternVideo /content/InternVideo   # Kaggle: /kaggle/working/InternVideo
pip install -q timm einops torchaudio huggingface_hub
export HF_TOKEN=...   # token của nguyenminh04 (đã chấp nhận điều khoản của các model gated)
python - <<'EOF'
from huggingface_hub import hf_hub_download
for repo, f in (('OpenGVLab/InternVideo2-Stage2_1B-224p-f4', 'InternVideo2-stage2_1b-224p-f4.pt'),
                ('OpenGVLab/InternVideo2-Stage2-6B-Audio', 'audio_6b.pth')):
    print(hf_hub_download(repo, f, local_dir='/content/iv2_ckpt'))
EOF
nohup python -u tools/coin_hub.py work --name colab-t4 --repo /content/InternVideo \
  --video-ckpt /content/iv2_ckpt/InternVideo2-stage2_1b-224p-f4.pt \
  --audio-ckpt /content/iv2_ckpt/audio_6b.pth --workers 4 > coin_worker.log 2>&1 &
```

- **Kaggle có 2×T4:** chạy hai tiến trình, mỗi tiến trình một GPU (`CUDA_VISIBLE_DEVICES=0` và `=1`), với hai `--name` khác nhau (`kaggle-t4-0`, `kaggle-t4-1`) và `--workers 2`, vì Kaggle chỉ có 4 vCPU.
- **Cần `ffmpeg` trên PATH.** Colab và Kaggle đều có sẵn.
- **Xem tiến độ:** `python tools/coin_hub.py status`, hoặc `tail coin_worker.log`.
- **Khi phiên bị ngắt:** chạy lại đúng lệnh đó với cùng `--name`. Máy sẽ nhận lại gói của chính nó, nhưng làm lại từ đầu gói (tối đa khoảng 40 phút).

## Sau khi xong

- Gộp `feats/*.tar` thành `data/coin/iv2_feats/`.
- Loại các video COIN trùng YouTube ID với tập val của YouCook2.
- Làm tiếp theo `docs/PLAN_multidataset.md`.
- Thư mục `videos/` trên HF (khoảng 95 GB) có thể xoá khi không cần nữa.
