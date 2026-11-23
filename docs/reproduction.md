# Ghi chú tái hiện

## 1. Đã kiểm chứng (offline, `pytest tests/`)
| Kiểm tra | Kết quả |
|---|---|
| Back-end Nes2Net-X / Nes2Net khớp code gốc | nạp `state_dict` gốc với `strict=True`, output **trùng bit-for-bit** (max diff = 0) |
| Số tham số back-end | Nes2Net-X 512,039 · Nes2Net 510,769 (paper: 511k) |
| MACs (4 s, 201 khung, chỉ conv/linear) | Nes2Net-X 84.2 M · Nes2Net 53.9 M (paper: 91.35 / 58.11; chênh lệch do hook không đếm BN, SE pooling và các phép nhân phần tử) |
| Chuyển key fairseq → HF | round-trip đủ mọi tensor, 0 missing / 0 unexpected, kể cả `weight_g/weight_v` và kiểu `parametrizations` của weight-norm |
| Front-end | 64,600 mẫu → 201 khung × 1024 |
| RawBoost algo 0–8 | chạy với numpy ≥ 2 (bản gốc lỗi `int(array)`, đã sửa đúng 1 dòng) |
| Pipeline đầy đủ | train → lưu → trung bình checkpoint → score → EER/minDCF/per-attack → cache đặc trưng → ablation (CPU, model tí hon) |

## 2. Khác biệt có chủ đích so với code gốc
| Mục | Code gốc | Repo này | Lý do |
|---|---|---|---|
| Front-end | fairseq `xlsr2_300m.pt` | HF `facebook/wav2vec2-xls-r-300m` (cùng trọng số, đã convert) | fairseq khó cài trên Python ≥ 3.10/Kaggle |
| Masking | `mask=False` | `mask_time_prob=0` | giữ nguyên hành vi |
| Layerdrop | theo cfg fairseq | 0.0 | ổn định; chỉ ảnh hưởng lúc train |
| Precision | fp32 | AMP fp16 (tuỳ chọn) | vừa T4 16 GB |
| Dữ liệu | thư mục LA gốc | HF Hub `Bisher/ASVspoof_2019_LA` (cùng 25,380 / 24,844 / 71,237 câu), hoặc thư mục gốc (`asv19dir:`) | Kaggle tải được trực tiếp |
| Tập test 2021/ITW | flac gốc + keys | parquet `SpeechAntiSpoofingBenchmarks/*` (DF đã re-encode FLAC, PCM giữ nguyên theo build script) | cùng lý do |

## 3. Công thức huấn luyện
* **Paper:** `lr=2.5e-7, batch 12, 100 epoch, RawBoost 4`; mỗi epoch với XLS-R trên T4 mất ~30–40 phút,
  tức ~60 giờ GPU, vượt xa một phiên Kaggle 12 giờ.
* **Kaggle budget (Notebook B):** `lr=1e-6`, dừng sau 8.3 giờ (~12–15 epoch), dev 6,000 câu, trung bình top-3.
  `lr=1e-6` là giá trị dùng cho wav2vec2-AASIST (SSL_Anti-spoofing) và cho bản CtrSVDD của chính tác giả.
* **Colab (Notebook `nes2net_colab.ipynb`):** `time_budget_h=3.5` mỗi phiên, dev 4,000 câu, resume từ Google Drive
  (`engine.train(resume=True, save_every_steps=300, keep_best_k=3)`). Colab free chỉ có 2 CPU, nên RawBoost trên
  CPU là nút cổ chai và epoch chậm hơn Kaggle.
* Đây **không** phải tái lập số liệu đầy đủ. Để so với bảng của paper, hãy dùng phần A (checkpoint chính thức)
  hoặc chạy `scripts/train.py --lr 2.5e-7 --epochs 100` trên GPU riêng.

## 4. Kết quả
Điền sau khi chạy notebook (các file `partA_results.json`, `partC_ablation.csv` trong `/kaggle/working`).

### 4.1 Checkpoint chính thức (`nes2net_x_DF1.65.pth`)
| Tập | Giao thức | EER (repo này) | Paper | Arena |
|---|---|---|---|---|
| ASVspoof2019 LA eval | 4 s | _ | — | 0.13 |
| ASVspoof2021 DF | phase=eval (10/80 shard) | _ | 1.65 | — |
| ASVspoof2021 DF | toàn bộ (10/80 shard) | _ | — | 3.61 (80/80) |
| In-the-Wild | 4 s đầu | _ | — | 8.48 |
| In-the-Wild | cả câu | _ | 5.52* | — |

\* checkpoint ITW tốt nhất của paper là một checkpoint khác (`Nes2Net-X w/o Val Aug`).

### 4.2 Huấn luyện lại (Kaggle budget)
| Tập | EER |
|---|---|
| ASVspoof2019 LA eval | _ |
| In-the-Wild | _ |
| DF eval-phase (10 shard) | _ |

### 4.3 Ablation (frozen XLS-R L24, mean ± std, 3 seed)
Xem `partC_ablation.csv` / `partC_ablation.png`.
