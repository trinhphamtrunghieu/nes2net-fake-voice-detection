# nes2net-fake-voice-detection

Tái hiện **Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-spoofing**
(Tianchi Liu, Duc-Tuan Truong, Rohan Kumar Das, Kong Aik Lee, Haizhou Li — IEEE T-IFS 2025,
[arXiv:2504.05657](https://arxiv.org/abs/2504.05657)) bằng PyTorch + 🤗 Transformers, **không cần fairseq**,
chạy được trên Kaggle, Colab hoặc máy local.

| | |
|---|---|
| 📓 Notebook (Kaggle · Colab · local) | [`notebooks/nes2net.ipynb`](notebooks/nes2net.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/trinhphamtrunghieu/nes2net-fake-voice-detection/blob/main/notebooks/nes2net.ipynb) [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/trinhphamtrunghieu/nes2net-fake-voice-detection/blob/main/notebooks/nes2net.ipynb) |
| 🤗 Demo | [HF Space](https://huggingface.co/spaces/trinhphamtrunghieu/nes2net-fake-voice-detection) · code trong `space/` |
| 📄 Tóm tắt paper | [`docs/paper_summary.md`](docs/paper_summary.md) |
| 🔬 Ablation, research gap, đề xuất | [`docs/ablation_and_gaps.md`](docs/ablation_and_gaps.md) |
| ✅ Ghi chú tái hiện & kết quả | [`docs/reproduction.md`](docs/reproduction.md) |

## Có gì trong repo
* `nes2net/backend.py`: Nes2Net và Nes2Net-X, **trùng bit-for-bit** với code gốc (kiểm bằng test), tên tham số
  giữ nguyên nên checkpoint chính thức nạp được `strict`. Có thêm công tắc ablation: tỉ lệ lồng, SE, pooling,
  dilation/kernel, bỏ kết nối tích luỹ ngoài/trong, cùng các baseline DR+Res2Net và mean+linear.
* `nes2net/model.py`: XLS-R/WavLM (HF) + back-end, và **bộ chuyển key fairseq → HF** để nạp checkpoint gốc.
* `nes2net/engine.py`: công thức train gốc (Adam, weighted CE [0.1, 0.9], RawBoost, lưu theo val loss có/không
  aug, trung bình checkpoint), cộng thêm AMP và giới hạn thời gian.
* `nes2net/ablation.py`: cache đặc trưng SSL đóng băng và lưới ablation nhiều seed.
* `nes2net/datasets_zoo.py`: ASVspoof2019 LA, In-the-Wild, ASVspoof2021 DF/LA lấy thẳng từ HF Hub hoặc từ
  thư mục gốc; lọc `phase=eval` như script chấm chính thức. `nes2net/full_eval.py`: chấm cả tập theo từng shard.
* `nes2net/presets.py`, `nes2net/env.py`, `nes2net/run.py`: cấu hình theo preset, nhận diện nền tảng, và các hàm
  chạy từng phần (A/B/C). Notebook và dòng lệnh `python -m nes2net.run` đều gọi chung các hàm này.
* `nes2net/logutil.py`: `train.log`, `metrics.jsonl`, TensorBoard.

## Dùng nhanh
```bash
pip install -r requirements.txt
pytest -q tests                       # khớp code gốc, convert key, pipeline

# đánh giá checkpoint chính thức
huggingface-cli download SpeechAntiSpoofingBenchmarks/Nes2Net nes2net_x_DF1.65.pth --local-dir ckpt
python scripts/evaluate.py --official ckpt/nes2net_x_DF1.65.pth \
    --sets hfbench:InTheWild asv19:test hfbench:ASVspoof2021_DF::eval

# train end-to-end (công thức paper: --lr 2.5e-7 --epochs 100)
python scripts/train.py --train asv19:train --dev asv19:validation --amp --grad_ckpt --out runs/x

# ablation back-end trên đặc trưng frozen
python scripts/ablate.py --groups A0_baselines A1_nest_ratio A5_connections --seeds 0 1 2
```

```python
from nes2net import load_official_checkpoint
m = load_official_checkpoint('ckpt/nes2net_x_DF1.65.pth').eval()
score = m(wave_4s_16k[None])[:, 1]     # cao hơn = bona fide
```

## Notebook: một file cho mọi nền tảng
`notebooks/nes2net.ipynb` tự nhận Kaggle / Colab / local và chọn chỗ lưu kết quả (`WORK`):
`/kaggle/working`, `MyDrive/nes2net` (Colab, mount Drive) hoặc `./work`. Dữ liệu và cache lớn nằm ở đĩa tạm.

1. Mở bằng badge ở trên. Kaggle: bật **GPU T4** và **Internet On**. Colab: *Runtime → Change runtime type → GPU*
   (Pro: A100/L4); ô Setup sẽ xin quyền mount Google Drive.
2. Trong ô Config chọn `PRESET`:

| Preset | Chạy gì | Thời gian (T4) |
|---|---|---|
| `smoke` | tập con rất nhỏ của mọi phần, chỉ để kiểm tra pipeline | ~30 phút |
| `default` | A (checkpoint chính thức trên tập con) + C (ablation) + C2 (chọn layer) | ~3–4 giờ |
| `full_eval` | A trên toàn bộ LA19 / ITW / 2021 LA / 2021 DF + C + C2 | ~6–8 giờ |
| `vi_zeroshot` | checkpoint chính thức, zero-shot trên tập thử tiếng Việt (FLEURS vi so với cùng câu do MMS-TTS đọc) + đối chứng tiếng Anh | ~30–45 phút |
| `paper_train` | B: train đúng công thức paper, resume qua nhiều phiên | ~80–100 giờ |

3. Ghi đè bất kỳ tham số nào qua `OVERRIDES`, ví dụ `{'B': {'lr': 2e-6}, 'run': {'L': False}}`.
   Cấu hình đầy đủ nằm ở `nes2net/presets.py`; chỉ `time_budget_h` thay đổi theo nền tảng.
4. **Train nhiều phiên:** chạy lại notebook với cùng preset là train tiếp từ `WORK/runB/last_state.pth`. Với các lần
   train dài, đặt `REPO_REF` thành một commit cố định để code không đổi giữa các phiên.

**Sửa code:** chỉ sửa trong `nes2net/` rồi push. Notebook clone repo lúc chạy nên không cần sửa notebook, trừ khi bạn
đổi tên hàm mà notebook gọi tới. Trên Colab, nếu runtime đã import `nes2net` thì phải *Restart session*.

**Dòng lệnh tương đương:** `python -m nes2net.run --preset full_eval --set run.L=false`

**Đẩy lên Kaggle bằng API:** `python scripts/kaggle_push.py --user <kaggle-user> --preset smoke`. Script chỉ thay dòng
`PRESET = ...` trong một bản sao tạm của notebook; với `paper_train`, thêm `--resume` từ phiên thứ hai để kernel dùng
output của phiên trước.

## Demo HF Space
Workflow `.github/workflows/deploy_space.yml` tự tạo hoặc cập nhật Space mỗi lần push vào `main` (cần
secret `HF_TOKEN` có quyền write; tên Space đặt qua biến `HF_SPACE`, mặc định `trinhphamtrunghieu/nes2net-fake-voice-detection`).
Deploy thủ công:
```bash
bash space/sync.sh                     # copy package vào space/
cd space && huggingface-cli upload trinhphamtrunghieu/nes2net-fake-voice-detection . --repo-type space
```
Space chạy trên **ZeroGPU** (Gradio 6, `@spaces.GPU`), theo hướng dẫn [huggingface-spaces skill](https://huggingface.co/new-space/agents.md). Workflow tự đặt hardware `zero-a10g` và biến `GRADIO_SSR_MODE=false`. Lần khởi động đầu cần tải checkpoint 1.3 GB.

## Ghi công
Kiến trúc, công thức huấn luyện và checkpoint thuộc về tác giả paper
([Nes2Net](https://github.com/Liu-Tianchi/Nes2Net), [Nes2Net_ASVspoof_ITW](https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW),
Apache-2.0). RawBoost lấy từ [SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing). Bản rehost
checkpoint và bộ parquet đánh giá do [SpeechAntiSpoofingBenchmarks](https://huggingface.co/SpeechAntiSpoofingBenchmarks) cung cấp.

```bibtex
@article{Nes2Net,
  author={Liu, Tianchi and Truong, Duc-Tuan and Das, Rohan Kumar and Lee, Kong Aik and Li, Haizhou},
  journal={IEEE Transactions on Information Forensics and Security},
  title={Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-Spoofing},
  year={2025}, volume={20}, pages={12005-12018}, doi={10.1109/TIFS.2025.3626963}}
```
