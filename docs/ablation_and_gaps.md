# Ablation có thể thực hiện, research gap và đề xuất

Tài liệu chia làm 3 phần: (I) các ablation, gồm cả những ablation paper chưa làm, kèm cách chạy trong repo;
(II) research gap rút ra từ paper, code gốc và các đánh giá độc lập; (III) phương pháp đề xuất cho từng gap,
kèm thiết kế thí nghiệm.

Ký hiệu: ✅ = paper đã làm · 🆕 = paper chưa làm, repo này đã cài sẵn công tắc · 🧪 = cần thêm code (đề xuất).

---

## I. Ablation

### I.1 Ablation back-end (rẻ, chạy trên đặc trưng SSL đóng băng: `nes2net/ablation.py`, Notebook phần C)

| ID | Câu hỏi | Biến thể | Trạng thái | Công tắc |
|---|---|---|---|---|
| A0 | Nes2Net có hơn baseline cùng ngân sách không? | mean+linear · DR(128)+Res2Net · Nes2Net · Nes2Net-X | ✅ (một phần) | `backend=` |
| A1 | Tỉ lệ lồng `[outer s1, inner s2]` | [4,4] [4,8] [8,4] **[8,8]** [8,16] [16,4] [16,8] [32,4] | 🆕 | `Nes_ratio` |
| A2 | SE có cần không, ratio bao nhiêu? | ratio 1/4/8/16, bỏ SE | 🆕 | `SE_ratio`, `use_se` |
| A3 | Pooling | **mean** vs ASTP (code gốc có sẵn ASTP nhưng paper không báo kết quả) | 🆕 | `pool_func` |
| A4 | Trường tiếp nhận thời gian | dilation 1/**2**/3, kernel 5 | 🆕 | `dilation`, `kernel_size` |
| A5 | **Cấu trúc "lồng" thật sự đóng góp gì?** | bỏ tích luỹ ngoài (`outer_hier=False`), bỏ tích luỹ trong (`inner_hier=False`), bỏ cả hai | 🆕 | `outer_hier`, `inner_hier` |
| A6 | Nes2Net vs -X với cùng MACs | so sánh ở cùng MACs (ví dụ Nes2Net `[8,16]` vs -X `[8,8]`) | 🆕 | kết hợp A1 |

A5 quan trọng nhất về mặt khoa học. Nếu bỏ cả hai kết nối tích luỹ, Nes2Net chỉ còn là một tập các nhóm
conv độc lập (gần với group convolution). Khi đó khoảng cách EER giữa "đầy đủ" và "bỏ cả hai" đo đúng phần
đóng góp của ý tưởng lồng nhau, tách khỏi tác dụng của việc "không có DR layer".

### I.2 Ablation front-end / huấn luyện (đắt, cần fine-tune end-to-end: `scripts/train.py`)

| ID | Câu hỏi | Biến thể | Trạng thái |
|---|---|---|---|
| B1 | Layer SSL nào? | layer 8/12/18/24, tổng có trọng số (`--ssl_layer weighted`) | 🆕 (bản frozen có ở Notebook C2) |
| B2 | Fine-tune hay đóng băng front-end? | full FT · frozen · 🧪 LoRA/adapter | 🆕 (`--freeze_ssl`) |
| B3 | RawBoost | algo 0 / 3 / 4 / 5 / 8 | ✅ (một phần: 4 vs 5 cho AASIST) |
| B4 | Augment lúc validation | có / không | ✅ |
| B5 | Trung bình checkpoint | 1 / 3 / 5 | ✅ |
| B6 | Độ dài cửa sổ train/test | 2 s / 4 s / 6 s; test 4 s đầu vs cả câu vs cửa sổ trượt | 🆕 (`--full`, demo có cửa sổ trượt) |
| B7 | Front-end khác | XLS-R 300M / 1B, WavLM-Large, HuBERT | 🆕 (`--ssl <hf id>`) |

### I.3 Ablation giao thức đánh giá (không cần train, Notebook phần A)

| ID | Câu hỏi | Ghi chú |
|---|---|---|
| E1 | ASVspoof 2021: phase `eval` hay toàn bộ trial? | Script chấm chính thức chỉ lấy `phase == eval`. Arena chấm toàn bộ 611,829 trial DF và báo **3.61%** cho đúng checkpoint mà paper báo **1.65%**; với LA là 6.14% vs 1.73%. |
| E2 | ITW: 4 s đầu hay cả câu? | Paper dùng `--test_protocol full` cho ITW; Arena dùng 4 s đầu (8.48%). |
| E3 | Phân rã theo tấn công / codec | `engine.evaluate_scores` trả EER theo `attack_id`; với DF có thể nhóm theo `codec`, `vocoder` (có trong `notes`). |

---

## II. Research gap

**G1. Chỉ dùng layer cuối của XLS-R.** Ở cấu hình ASVspoof, back-end chỉ nhận `last_hidden_state`. Nhiều công
trình trước đó cho thấy thông tin phát hiện giả mạo tập trung ở các layer giữa của SSL; chính bản WavLM của
tác giả (CtrSVDD) cũng gộp 25 layer bằng SEA. Paper không so sánh việc chọn layer cho wav2vec2.

**G2. Nhóm kênh cố định, liên kết một chiều.** Outer split cắt 1024 kênh thành 8 khối *liền nhau* theo chỉ số
kênh, và thông tin chỉ chảy một chiều `x_1 → x_2 → … → x_7`; `x_8` không qua xử lý gì. Thứ tự kênh của SSL không
mang ý nghĩa, nên việc gán kênh vào nhóm là tuỳ ý, và các nhóm đầu không nhìn thấy nhóm sau. Bằng chứng gián
tiếp: MultiAPI-Spoof (arXiv:2512.07352) đã thêm local cross-block attention giữa các khối lồng nhau của Nes2Net-X.

**G3. Mô hình hoá thời gian ngắn hạn, pooling đơn giản.** Toàn bộ conv thời gian có `k=3, d=2` (trường tiếp
nhận vài khung mỗi khối), sau đó là mean pooling. Cách này bất lợi cho partial spoof (đoạn giả ngắn bị trung
bình hoá) và cho các dấu vết dài hạn như prosody hay nhịp.

**G4. "Nhẹ" chỉ ở back-end.** Back-end có 0.5M tham số / ~58–91 MMACs, nhưng XLS-R 300M có ~317M tham số và
hàng chục GMACs cho mỗi 4 s, nên cả hệ thống không nhẹ. Hơn nữa, việc fine-tune toàn bộ 300M tham số trên
ASVspoof2019 LA (chỉ 20 người nói, 6 tấn công) dễ overfit.

**G5. Tổng quát hoá ra ngoài miền còn yếu.** Đánh giá độc lập của cùng checkpoint trên Arena cho EER: CD-ADD
20.6%, ASVspoof5 22.3%, SONAR 33.3%, ADD22 26.6%, ODSS 29.9%. Với TTS hiện đại, ngôn ngữ khác và codec mới,
EER tăng 5–20 lần so với ASVspoof2021. Paper chỉ đánh giá trên các tập "cổ điển".

**G6. Độ nhạy với giao thức và khả năng tái lập.** Cùng một checkpoint, EER thay đổi khoảng 2–4 lần chỉ vì cách
chọn trial (E1) và cách cắt cửa sổ (E2). Kết quả trong paper là "best (mean)" qua seed và có chọn epoch theo
validation. Khi so sánh với các bảng khác, cần ghi rõ giao thức.

**G7. Chi phí của Nes2Net-X.** Weighted sum làm MACs tăng 57% và độ trễ tăng gần gấp đôi (9.2 vs 4.9 ms), trong
khi EER chỉ giảm 0.05 điểm trên CtrSVDD. Trên ASVspoof5, Nes2Net thường (6.13) còn không hơn AASIST (6.08) về EER.

**G8. Điểm số chưa hiệu chỉnh, không có độ bất định.** Đầu ra là logit; paper chỉ báo EER (và CLLR/minDCF cho
ASVspoof5). Ngưỡng quyết định khi triển khai, chẳng hạn trong demo, không chuyển được giữa các miền dữ liệu.

**G9. Tấn công khó (A12, singing).** Tác giả tự nêu A12 là điểm yếu chung của mọi mô hình.

---

## III. Đề xuất phương pháp

### P1 (cho G1): Layer–Channel Nested Aggregation ("Nes3Net")
Thêm **trục layer** vào cấu trúc lồng. Với `H ∈ R^{L×C×T}` (L layer), nhóm layer thành `g` cụm (ví dụ 4 cụm
6 layer), mỗi cụm gộp bằng softmax-weighted sum thành đầu vào cho một nhánh outer. Nhánh outer `i` khi đó nhận
`x_i^{(cụm i)} + y_{i-1}`: nhóm đầu thấy layer nông, nhóm sau thấy layer sâu hơn kèm thông tin đã tích luỹ từ
các nhóm trước, tức là tổ chức đa thang đo theo **độ sâu biểu diễn** chứ không chỉ theo trường tiếp nhận.
* Chi phí thêm: L tham số mỗi cụm, gần như 0 MACs.
* Baseline: chỉ layer cuối · `LayerMix` toàn cục (đã cài, `--ssl_layer weighted`) · SEA của tác giả.
* Thí nghiệm: Notebook C2 (frozen) trước, sau đó fine-tune với lr nhỏ.

### P2 (cho G2): Learnable channel grouping + bidirectional nesting
1. **Hoán vị kênh học được / theo cụm:** trước outer split, sắp lại kênh theo phân cụm (k-means trên hệ số tương
   quan kênh của đặc trưng SSL, tính một lần), hoặc thêm một Conv1×1 nhỏ dạng *block-diagonal shuffle*.
2. **Lồng hai chiều:** chạy chuỗi outer theo cả hai hướng (`1→8` và `8→1`) với trọng số dùng chung, rồi gộp lại,
   để mọi nhóm đều nhìn thấy mọi nhóm khác. Có thể so với cross-block attention (MultiAPI-Spoof) ở cùng ngân sách.
3. Xử lý luôn nhóm cuối (bỏ passthrough `x_8`) và kiểm tra lại bằng A5.
* Đo: EER ITW / DF, MACs. Kỳ vọng: cải thiện ở ngoài miền với ≤ +10% MACs.

### P3 (cho G3): Multi-dilation nesting + attentive / frame-level heads
* Mỗi nhánh outer `i` dùng dilation `d_i ∈ {1,2,4,8,…}`, tức đa thang đo về thời gian song song với đa thang đo
  về kênh, ở cùng số tham số.
* Thay mean pooling bằng ASTP (A3) hoặc **max + mean** để không bỏ sót đoạn giả ngắn.
* Với PartialSpoof: thêm đầu ra mức khung (bỏ pooling, FC từng khung) và huấn luyện đa nhiệm câu + khung.

### P4 (cho G4): Front-end hiệu quả tham số
* **LoRA (r=8) hoặc adapter** trên q/v của XLS-R thay cho fine-tune toàn bộ (~1–2M tham số train được), kết hợp
  Nes2Net. Giả thuyết: ít overfit ASVspoof2019 hơn nên tổng quát tốt hơn (G5).
* **Cắt layer:** chỉ giữ 12–16 layer đầu (theo kết quả P1/B1), giảm khoảng 40% MACs front-end.
* **Distillation:** student WavLM-Base+ / DistilHuBERT + Nes2Net, teacher là XLS-R + Nes2Net-X.
* Báo cáo **MACs toàn hệ thống**, không chỉ của back-end.

### P5 (cho G5, G9): Dữ liệu và mục tiêu huấn luyện
* **Codec-aware augmentation:** ngoài RawBoost, thêm MP3/AAC/Opus/EnCodec ngẫu nhiên (DF và ITW chứa nhiều codec).
* **Huấn luyện đa nguồn:** thêm MLAAD, ASVspoof5 train, CodecFake, với phân chia leave-one-dataset-out để đo
  tổng quát hoá một cách trung thực.
* **One-class / SAM:** OC-Softmax hoặc AM-Softmax để nén vùng bona fide; SAM/ASAM để tìm điểm cực tiểu phẳng.

### P6 (cho G6, G8): Giao thức và hiệu chỉnh
* Luôn báo cả 3 con số: `phase=eval` (so với paper), toàn bộ trial (so với benchmark), và theo codec/attack.
* **Cửa sổ trượt + gộp điểm** (mean / max / trimmed-mean) thay cho cửa sổ 4 s đầu. Demo đã có chế độ này.
* **Hiệu chỉnh:** temperature scaling hoặc logistic calibration trên một tập dev ngoài miền; báo CLLR,
  actDCF và ECE.

### P7 (cho G7): Nes2Net-X rẻ hơn
Weighted sum đang làm Conv2d chạy trên `(i+2)` bản sao. Có thể thay bằng: (a) tổ hợp tuyến tính *trước* conv
(`conv(Σ w_j s_j)`), vốn tương đương toán học khi không có phi tuyến giữa hai bước, nên MACs gần bằng
Nes2Net; hoặc (b) trọng số dùng chung giữa các khối. Cần kiểm tra bằng thực nghiệm phần lợi ích còn giữ được.
Lưu ý: ReLU và BN nằm giữa conv và tổng nên (a) không tương đương hoàn toàn, đây chính là điều cần đo.

---

## IV. Thứ tự ưu tiên gợi ý
1. **E1/E2** (miễn phí, chỉ cần checkpoint có sẵn): xác lập giao thức.
2. **A1–A5 + B1** trên đặc trưng frozen (vài giờ trên Kaggle): xác định thành phần nào thực sự quan trọng.
3. **P1 + P4 (LoRA)**: hai đề xuất rẻ nhất có khả năng cải thiện ngoài miền; thử end-to-end 1 seed trước, 3 seed sau.
4. **P5** nếu có thêm dữ liệu và GPU.

Mọi kết quả nên báo dạng mean ± std trên ≥ 3 seed, kèm MACs/params của cả hệ thống.
