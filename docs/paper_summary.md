# Tóm tắt paper: Nes2Net (arXiv:2504.05657, IEEE T-IFS 2025)

**Tác giả:** Tianchi Liu, Duc-Tuan Truong, Rohan Kumar Das, Kong Aik Lee, Haizhou Li
**Code gốc:** [Liu-Tianchi/Nes2Net](https://github.com/Liu-Tianchi/Nes2Net) (CtrSVDD, WavLM) ·
[Liu-Tianchi/Nes2Net_ASVspoof_ITW](https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW) (ASVspoof 2019/2021, In-the-Wild; nhánh `asvspoof5`)

## 1. Vấn đề
Các hệ chống giả mạo giọng nói hiện đại = *speech foundation model* (wav2vec 2.0 XLS-R, WavLM; đặc trưng
1024 chiều/khung) + *back-end* phân loại. Back-end phổ biến (AASIST, ResNet, ECAPA-TDNN…) được thiết kế cho
đặc trưng chiều thấp nên phải chèn **lớp giảm chiều (DR layer)** 1024 → 128. Paper chỉ ra DR layer:
1. tốn tham số/tính toán (một Linear 1024×128 đã là 131k tham số);
2. có thể làm mất thông tin hữu ích trong không gian chiều cao;
3. tăng chiều D không giúp được (D=256 còn tệ hơn D=128 do over-parameterization).

## 2. Phương pháp
### 2.1 Res2Net không DR (ResNet-woDR)
Thử nghiệm đầu: bỏ DR, dùng Res2Net trực tiếp trên 1024 kênh với *scale* `s` lớn để mỗi nhánh nhỏ đi.
Tốt nhất ở `s = 64` (EER 2.73% trên CtrSVDD), nhưng tăng `s` hơn nữa hoặc tăng độ sâu không giúp → cần một
cấu trúc khác để vừa nhẹ vừa tương tác đa thang đo.

### 2.2 Nested Res2Net (Nes2Net)
Hai tầng Res2Net lồng nhau (code: `Nes_ratio=[8, 8]`):

```
x ∈ R^{1024×T}  ──split──>  x_1 … x_8   (mỗi nhóm 128 kênh)          [outer, s1 = 8]
y_1 = BN(ReLU(B_1(x_1)))
y_i = BN(ReLU(B_i(x_i + y_{i-1})))      i = 2..7    (tích luỹ kiểu Res2Net)
y_8 = x_8                                (nhóm cuối đi thẳng)
z   = ReLU(BN(concat(y_1..y_8)))  →  mean_T  →  FC(1024 → 2)
```
Mỗi `B_i` là một **Bottle2neck** (khối Res2Net bên trong, `s2 = 8`):
`Conv1×1 → split 8×16 kênh → chuỗi Conv1d(k=3, dilation=2) có kết nối tích luỹ → Conv1×1 → SE(ratio 1) → + residual`.

### 2.3 Nes2Net-X
Ở tầng trong, thay phép **cộng** `sp + x_i` bằng **ghép** theo một trục mới rồi dùng `Conv2d (k×1)` và
**tổng có trọng số học được** (khởi tạo đều `1/(i+2)`). Cùng số tham số (~511k) nhưng MACs tăng
58 → 91 M, độ chính xác tăng nhẹ.

### 2.4 Huấn luyện (theo code chính thức)
| | ASVspoof/ITW (wav2vec2 XLS-R 300M) | CtrSVDD (WavLM-Large) |
|---|---|---|
| Front-end | fine-tune toàn bộ, layer cuối | fine-tune, gộp 25 layer bằng SEA |
| Input | 64,600 mẫu đầu (~4 s), lặp nếu ngắn | 4 s |
| Augment | RawBoost algo 4 (LnL+ISD+SSI nối tiếp) | RawBoost algo 8 |
| Optim | Adam, lr 2.5e-7, wd 1e-4, batch 12, 100 epoch | lr 1e-6, batch 34, 75 epoch |
| Loss | Weighted CE [0.1 spoof, 0.9 bona fide] | |
| Chọn model | val loss (có/không aug), trung bình 3–5 checkpoint | |

## 3. Kết quả chính (EER %, best (mean))
**CtrSVDD (Table III, WavLM):**

| Back-end | Params | MMACs | EER |
|---|---|---|---|
| AASIST (C=32) | 447k | 707.65 | 3.30 (3.36) |
| ResNet | 611k | 70.62 | 3.11 (3.61) |
| Res2Net (DR=128) | 452k | 64.93 | 3.02 (3.24) |
| ECAPA-TDNN | 497k | 80.21 | 3.19 (3.74) |
| **Nes2Net** | 511k | **58.11** | 2.53 (2.55) |
| **Nes2Net-X** | 511k | 91.35 | **2.48 (2.51)** |

**ASVspoof 2021 (wav2vec2, train ASVspoof2019 LA):** Nes2Net-X LA 1.73 (1.95), DF 1.65 (1.91); trung bình 5
checkpoint DF **1.49 (1.78)**. AASIST cùng front-end: LA 0.82, DF 2.85 → Nes2Net tốt hơn rõ ở DF, kém hơn ở LA.
**In-the-Wild:** Nes2Net-X 5.52 (6.60); có val-aug 5.15 (6.31).
**PartialSpoof:** Nes2Net eval 0.53 (0.68). **ASVspoof 5:** Nes2Net-X EER 5.92 vs AASIST 6.08 (Nes2Net thường 6.13 —
không hơn AASIST về EER; lợi thế chủ yếu ở minDCF/CLLR và chi phí tính toán).

## 4. Ablation có trong paper
* Res2Net + DR: scale `s` (4/16/64/128) và chiều D (64/128/256)
* ResNet-woDR: scale `s` (4…256), độ sâu (b = 1/2/4)
* Nes2Net vs Nes2Net-X; augment lúc validation; trung bình checkpoint; scale của AASIST

**Không có ablation** cho: tỉ lệ lồng `[outer, inner]`, SE ratio, pooling (mean vs ASTP), dilation/kernel,
vai trò riêng của kết nối tích luỹ ngoài/trong, và việc chọn layer SSL → repo này bổ sung các ablation đó
(xem `ablation_and_gaps.md`).

## 5. Hạn chế tác giả tự nêu
* Tấn công A12 (singing voice synthesis) khó nhất với mọi mô hình.
* Nes2Net-X chính xác hơn nhưng chậm hơn (9.2 vs 4.9 ms/batch).
* Có nên augment lúc validation hay không vẫn là câu hỏi mở.
