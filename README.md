# nes2net-fake-voice-detection

Tái hiện **Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-spoofing**
(Tianchi Liu, Duc-Tuan Truong, Rohan Kumar Das, Kong Aik Lee, Haizhou Li — IEEE T-IFS 2025,
[arXiv:2504.05657](https://arxiv.org/abs/2504.05657)) bằng PyTorch + 🤗 Transformers, **không cần fairseq**,
chạy được trên Kaggle.

| | |
|---|---|
| 📓 Kaggle notebook | [`notebooks/nes2net_kaggle.ipynb`](notebooks/nes2net_kaggle.ipynb) · [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/trinhphamtrunghieu/nes2net-fake-voice-detection/blob/main/notebooks/nes2net_kaggle.ipynb) |
| 📓 Colab notebook | [`notebooks/nes2net_colab.ipynb`](notebooks/nes2net_colab.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/trinhphamtrunghieu/nes2net-fake-voice-detection/blob/main/notebooks/nes2net_colab.ipynb) |
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
  thư mục gốc; lọc `phase=eval` như script chấm chính thức.

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

## Kaggle
1. Bấm badge *Open in Kaggle* ở trên (hoặc *File → Import Notebook* → dán URL của `notebooks/nes2net_kaggle.ipynb`).
2. Settings: **GPU T4/P100**, **Internet On**.
3. Chỉnh `RUN_A / RUN_B / RUN_C` trong ô Config → *Save & Run All*.

## Google Colab
1. Bấm badge *Open In Colab* ở trên → *Runtime → Change runtime type → GPU*.
2. Chạy ô Setup: cho phép **mount Google Drive**. Kết quả, checkpoint và trạng thái resume nằm ở
   `MyDrive/nes2net`; dữ liệu và cache nằm ở `/content` (tự tải lại mỗi phiên).
3. Chỉnh `RUN_A / RUN_B / RUN_C` → *Runtime → Run all*.
4. Phần B (train end-to-end) chạy theo từng phiên (`time_budget_h=3.5`) và lưu `last_state.pth` mỗi 300 bước. Khi
   Colab ngắt kết nối, chỉ cần chạy lại notebook là train tiếp. Mặc định giữ 3 checkpoint tốt nhất (~8 GB trên
   Drive; Drive free có 15 GB).

Hai notebook được sinh từ cùng một nguồn (`notebooks/build_notebook.py` + `notebooks/_cells.py`); sửa ở đó rồi
chạy `python notebooks/build_notebook.py`.

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

## Kaggle smoke run (kiểm tra pipeline nhanh, ~30–45 phút)
`kaggle/nes2net_kaggle_smoke.ipynb` chạy cả 3 phần A/B/C trên tập con rất nhỏ (vài trăm câu), chỉ để xác nhận
code chạy được trên Kaggle GPU. Số liệu của nó không có giá trị so sánh.
```bash
sed -i "s/__KAGGLE_USER__/<kaggle-username>/" kaggle/kernel-metadata.json
kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4
kaggle kernels status <kaggle-username>/nes2net-smoke-run
kaggle kernels output <kaggle-username>/nes2net-smoke-run -p kaggle_out
```
