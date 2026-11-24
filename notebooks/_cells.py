# Shared experiment cells (exec-ed by build_notebook.py with md/code/WORKDIR_NOTE defined).
md("## 0. Kiểm tra nhanh: back-end khớp bit-for-bit với code gốc, số tham số ≈ 511k")
code(r"""
r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', os.path.join(REPO_DIR, 'tests/test_all.py')], capture_output=True, text=True)
print(r.stdout[-1500:])
from nes2net.backend import build_backend, count_params, count_macs
for n in ['nes2net', 'nes2net_x', 'dr_res2net']:
    m = build_backend(n)
    print(f'{n:12s} params={count_params(m)/1e3:7.1f}k  MMACs(4s)={count_macs(m):6.1f}')
print('Paper (Table III): Nes2Net 511k / 58.11 MMACs, Nes2Net-X 511k / 91.35 MMACs')
""")

md(r"""
## A. Checkpoint chính thức → đánh giá
`load_official_checkpoint` chuyển key fairseq → HF `Wav2Vec2Model` (không cần cài fairseq) và nạp **strict**
(không thiếu/thừa key nào), nên mọi trọng số đều là của tác giả.

Lưu ý giao thức: script chấm chính thức của ASVspoof 2021 chỉ tính EER trên các trial có `phase == eval`;
một số benchmark (ví dụ Speech Anti-Spoofing Arena) tính trên **toàn bộ** 611,829 trial DF (gồm progress và
hidden-track) và báo 3.61% cho cùng checkpoint mà paper báo 1.65%. Ô dưới chấm cả hai cách.
""")
code(r"""
RESULTS = {}
if RUN_A:
    from huggingface_hub import hf_hub_download
    from nes2net.model import load_official_checkpoint
    from nes2net.datasets_zoo import get_items
    from nes2net import engine
    ck = hf_hub_download('SpeechAntiSpoofingBenchmarks/Nes2Net', 'nes2net_x_DF1.65.pth')
    model = load_official_checkpoint(ck).to(DEV)
    print('loaded official Nes2Net-X (strict).')
    sets = dict(A_SETS)
    if A_ITW_FULL:
        sets['ITW (full utterance)'] = ('hfbench:InTheWild', True)
    cache = {}
    for name, (spec, full) in sets.items():
        items = cache.setdefault(spec, get_items(spec))
        s = engine.score(model, items, bs=1 if full else 32, device=DEV, full=full, workers=NW, desc=name)
        engine.write_scores(s, f'{WORK}/A_{name.replace(" ", "_").replace("/", "-")}.txt')
        RESULTS[name] = engine.evaluate_scores(s, items)
        print(f"{name:48s} n={RESULTS[name]['n']:7d}  EER={RESULTS[name]['eer']:.2f}%  minDCF={RESULTS[name]['minDCF']:.4f}")
    json.dump(RESULTS, open(f'{WORK}/partA_results.json', 'w'), indent=1)
    del model; torch.cuda.empty_cache()
""")
code(r"""
if RUN_A:
    ref = {'ITW (4s first window)': 'paper (other ckpt, full utt): 5.52 best / 6.60 mean; Arena same ckpt: 8.48',
           'ITW (full utterance)': 'paper: Nes2Net-X 5.52 (best) / 6.60 (mean), different ckpt',
           'ASVspoof2019 LA eval': 'Arena same ckpt: 0.13',
           'ASVspoof2021 DF (phase=eval, 10/80 shards)': 'paper: 1.65 (this ckpt)',
           'ASVspoof2021 DF (all trials, 10/80 shards)': 'Arena (all 611,829 trials): 3.61'}
    display(pd.DataFrame([{'set': k, 'n': v['n'], 'EER %': round(v['eer'], 2), 'minDCF': round(v['minDCF'], 4),
                           'reference': ref.get(k, '')} for k, v in RESULTS.items()]))
    for k, v in RESULTS.items():
        if 'per_attack_eer' in v:
            print(k, {a: round(e, 2) for a, e in v['per_attack_eer'].items()})
""")

md(r"""
## B. Huấn luyện end-to-end (công thức của paper, rút gọn theo ngân sách GPU notebook)
Giống code gốc: XLS-R 300M fine-tune toàn bộ + Nes2Net-X `[8,8]`, SE ratio 1, dilation 2, mean pooling,
Adam (wd 1e-4), weighted CE `[0.1, 0.9]`, RawBoost algo 4, cửa sổ 4 s đầu, lưu checkpoint khi val loss
(có/không aug) giảm, rồi lấy trung bình top-k checkpoint.

Khác paper (do giới hạn phiên): `lr=1e-6` thay vì `2.5e-7` (100 epoch không chạy kịp), AMP fp16,
dev 4,000–6,000 câu thay vì 24,844, dừng theo thời gian.

**Resume:** trạng thái (model + Adam + history) được ghi vào `WORK/runB/last_state.pth` mỗi `save_every_steps` bước
và cuối mỗi epoch. Bị ngắt kết nối/hết giờ → chạy lại notebook với `RUN_B = True`, huấn luyện tiếp từ chỗ dừng.
Chỉ giữ `keep_best_k` checkpoint tốt nhất để tiết kiệm dung lượng. Đánh giá (trung bình top-k checkpoint, chấm
LA19 / ITW / DF, ~1 giờ) chỉ chạy khi `B_EVAL = True` — bật khi bạn đã train đủ số epoch mong muốn.
""")
code(r"""
if RUN_B:
    from nes2net.model import SSLNes2Net
    from nes2net.datasets_zoo import get_items
    from nes2net import engine
    engine.set_seed(12345)
    model = SSLNes2Net('facebook/wav2vec2-xls-r-300m', backend='nes2net_x', gradient_checkpointing=True)
    tr, dv = get_items(globals().get('B_TRAIN', 'asv19:train')), get_items(B['dev'])
    out = f'{WORK}/runB'
    hist = engine.train(model, tr, dv, out, epochs=B['epochs'], lr=B['lr'], bs=B['bs'], algo=B['algo'],
                        device=DEV, amp=True, workers=NW, time_budget_h=B['time_budget_h'],
                        resume=True, keep_best_k=B['keep_best_k'], save_every_steps=B['save_every_steps'])
    print(f"đã xong {len(hist)}/{B['epochs']} epoch; checkpoint & last_state.pth ở {out}")
if RUN_B and B_EVAL:
    best = engine.best_epochs(hist, k=B['avg_k']); print('best epochs (val loss w/o aug):', best)
    assert best, 'chưa có epoch nào hoàn tất — chạy tiếp phần train trước'
    engine.average_checkpoints(model, [f'{out}/epoch_{e}.pth' for e in best], f'{out}/avg.pth')
    model.to(DEV)
    for name, spec in globals().get('B_EVAL_SETS', [('ASVspoof2019 LA eval', 'asv19:test'), ('ITW', 'hfbench:InTheWild'),
                                                    ('DF eval-phase 10 shards', 'hfbench:ASVspoof2021_DF:10:eval')]):
        items = get_items(spec); s = engine.score(model, items, device=DEV, workers=NW, desc=name)
        r = engine.evaluate_scores(s, items); print(name, round(r['eer'], 2)); RESULTS['B ' + name] = r
    json.dump(RESULTS, open(f'{WORK}/partB_results.json', 'w'), indent=1)
    pd.DataFrame(hist).plot(x='epoch', y=['train_loss', 'val_loss_wo_aug', 'val_loss_w_aug'], logy=True)
""")

md(r"""
### Theo dõi log của phần B
`engine.train` ghi ba loại log vào `WORK/runB`, và lần resume nối tiếp vào đúng các file đó:
* `train.log`: log dạng chữ, giống console. Cứ `log_every=50` bước ghi một dòng gồm loss, lr, utt/s, % thời gian
  chờ DataLoader (nếu cao thì RawBoost/CPU là nút cổ chai), GPU memory, ETA của epoch; cuối mỗi epoch ghi val loss và val EER.
* `metrics.jsonl`: cùng các số đó dạng JSON (event `start` / `train_step` / `epoch` / `checkpoint` / `resume` / `stop`).
* `tb/`: TensorBoard (đặt `SHOW_TB = True` để mở ngay trong notebook).
""")
code(r"""
SHOW_TB = False
_mp = f'{WORK}/runB/metrics.jsonl'
if os.path.exists(_mp):
    from nes2net.logutil import load_metrics
    import matplotlib.pyplot as plt
    M = load_metrics(_mp)
    st, ep = M[M.event == 'train_step'], M[M.event == 'epoch']
    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    if len(st):
        x = (st.epoch - 1) + st.step / st.steps
        ax[0].plot(x, st.loss, lw=.7, label='train loss (window)'); ax[0].set_xlabel('epoch')
        ax[1].plot(x, st.utt_per_s, lw=.7, label='utt/s'); ax[1].plot(x, 100 * st.data_frac, lw=.7, label='% chờ dữ liệu')
    if len(ep):
        ax[0].plot(ep.epoch, ep.val_loss_wo_aug, 'o-', label='val loss'); ax[0].plot(ep.epoch, ep.val_loss_w_aug, 's-', label='val loss (aug)')
        ax[2].plot(ep.epoch, ep.val_eer_wo_aug, 'o-', label='val EER %'); ax[2].plot(ep.epoch, ep.val_eer_w_aug, 's-', label='val EER % (aug)')
    for a in ax: a.legend(); a.grid(alpha=.3)
    plt.tight_layout(); plt.show()
    print(open(f'{WORK}/runB/train.log').read()[-3000:])
    if SHOW_TB:
        get_ipython().run_line_magic('load_ext', 'tensorboard')
        get_ipython().run_line_magic('tensorboard', f'--logdir {WORK}/runB/tb')
""")

md(r"""
## C. Ablation back-end trên đặc trưng XLS-R đóng băng
Mỗi câu được cắt 4 s đầu, đặc trưng layer cuối XLS-R (pre-trained, **không fine-tune**) được cache fp16 một lần;
sau đó mỗi back-end chỉ train vài phút. EER tuyệt đối cao hơn paper (SSL đóng băng, tập con) —
chỉ so sánh **tương đối** giữa các biến thể, mỗi cấu hình chạy 3 seed (mean ± std).

Nhóm ablation (chi tiết và giả thuyết trong `docs/ablation_and_gaps.md`):
* **A0** baseline: mean+linear, DR(128)+Res2Net, Nes2Net, Nes2Net-X
* **A1** tỉ lệ lồng `[outer, inner]` — paper **không** ablate
* **A2** SE ratio / bỏ SE
* **A3** mean pooling vs ASTP
* **A4** dilation / kernel
* **A5** bỏ kết nối tích luỹ ngoài (outer) / trong (inner) — kiểm chứng trực tiếp đóng góp của cấu trúc "lồng"
""")
code(r"""
if RUN_C:
    from transformers import AutoModel
    from nes2net.datasets_zoo import get_items
    from nes2net import ablation
    tag = lambda s: s.replace(':', '_').replace('/', '_').replace('@', '_n')
    ssl = AutoModel.from_pretrained('facebook/wav2vec2-xls-r-300m')
    dirs = {s: ablation.extract_features(ssl, get_items(s), f'{TMP}/feat/{tag(s)}', layers=C['layers'], device=DEV, workers=NW)
            for s in [C['train'], C['dev']] + C['evals']}
    del ssl; torch.cuda.empty_cache()
    dfC = ablation.run_grid(C['groups'], dirs[C['train']], dirs[C['dev']], {tag(s): dirs[s] for s in C['evals']},
                            seeds=C['seeds'], out_csv=f'{WORK}/partC_ablation.csv', layers=C['layers'],
                            epochs=C['epochs'], device=DEV)
    display(ablation.summarize(dfC))
""")
code(r"""
if RUN_C:
    import matplotlib.pyplot as plt
    evc = [c for c in dfC.columns if c.endswith('_eer') and c != 'dev_eer']
    fig, axes = plt.subplots(1, len(evc), figsize=(7 * len(evc), 6))
    axes = np.atleast_1d(axes)
    lab = dfC['group'].str[:2] + ' ' + dfC['cfg'].str.replace('"backend": ', '').str.replace('"', '')
    for ax, c in zip(axes, evc):
        g = dfC.assign(lab=lab).groupby('lab')[c]
        m, s = g.mean(), g.std()
        ax.barh(m.index, m.values, xerr=s.values, color='#4C72B0'); ax.set_title(c + ' (mean ± std, 3 seeds)')
        ax.set_xlabel('EER %'); ax.invert_yaxis()
    plt.tight_layout(); plt.savefig(f'{WORK}/partC_ablation.png', dpi=120); plt.show()
""")

md(r"""
### C2. Chọn layer SSL (research gap G1)
Paper dùng **layer cuối** của XLS-R. Nhiều nghiên cứu cho thấy thông tin phát hiện giả mạo nằm ở các layer
giữa. Ở đây cache 4 layer trên tập nhỏ hơn và so sánh từng layer với tổng có trọng số học được (`LayerMix`).
""")
code(r"""
if RUN_C and C_LAYER_STUDY:
    LAY = [8, 12, 18, 24]   # 4 layer x 6k câu ~ 10 GB cache
    ssl = AutoModel.from_pretrained('facebook/wav2vec2-xls-r-300m')
    small = {'train': f"asv19:train@{C_LAYER_N['train']}", 'dev': f"asv19:validation@{C_LAYER_N['dev']}", 'itw': f"hfbench:InTheWild@{C_LAYER_N['itw']}"}
    d2 = {k: ablation.extract_features(ssl, get_items(s), f'{TMP}/featL/{tag(s)}', layers=LAY, device=DEV, workers=NW) for k, s in small.items()}
    del ssl; torch.cuda.empty_cache()
    rows = []
    for layers in [[l] for l in LAY] + [LAY]:
        for sd in C['seeds']:
            r = ablation.train_backend({'backend': 'nes2net_x'}, d2['train'], d2['dev'], {'itw': d2['itw']},
                                       layers=layers, seed=sd, epochs=C['epochs'], device=DEV)
            r['layers'] = 'mix' + str(LAY) if len(layers) > 1 else f'L{layers[0]}'; rows.append(r); print(r['layers'], sd, round(r['itw_eer'], 2))
    dfL = pd.DataFrame(rows); dfL.to_csv(f'{WORK}/partC_layers.csv', index=False)
    display(dfL.groupby('layers')[['dev_eer', 'itw_eer']].agg(['mean', 'std']).round(2))
""")
md(r"""
## Kết quả lưu tại thư mục `WORK`
(Kaggle: `/kaggle/working`; Colab: `MyDrive/nes2net` nếu mount Drive)
`partA_results.json`, `partB_results.json`, `partC_ablation.csv`, `partC_layers.csv`, `partC_ablation.png`, các file score.
Copy các con số vào `docs/results.md` của repo.
""")

