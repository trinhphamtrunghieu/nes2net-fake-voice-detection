"""Generates notebooks/nes2net_kaggle.ipynb and notebooks/nes2net_colab.ipynb.

Run: python notebooks/build_notebook.py   (both notebooks share all experiment cells; only the
header, setup and config cells differ by platform)
"""
import nbformat as nbf
import os

REPO_URL = os.environ.get('REPO_URL', 'https://github.com/trinhphamtrunghieu/nes2net-fake-voice-detection.git')
BLOB = 'github/trinhphamtrunghieu/nes2net-fake-voice-detection/blob/main/notebooks/nes2net_colab.ipynb'

INTRO = r"""
**Paper:** *Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-spoofing*
(Liu, Truong, Das, Lee, Li — IEEE T-IFS 2025, [arXiv:2504.05657](https://arxiv.org/abs/2504.05657)).

Notebook gồm 3 phần độc lập (bật/tắt trong ô **Config**):

| Phần | Nội dung | Thời gian (T4) |
|---|---|---|
| **A** | Nạp checkpoint chính thức Nes2Net-X (wav2vec2 XLS-R) → đánh giá ITW / ASVspoof2019 LA / ASVspoof2021 DF, so với paper | ~30–60 phút |
| **B** | Huấn luyện lại end-to-end (XLS-R fine-tune + Nes2Net-X, RawBoost) trên ASVspoof2019 LA, có **resume** | nhiều phiên |
| **C** | Ablation back-end trên đặc trưng XLS-R đóng băng (nhiều cấu hình × 3 seed) | ~1–3 giờ |

Dữ liệu được tải từ Hugging Face Hub: `Bisher/ASVspoof_2019_LA` (train/val/test),
`SpeechAntiSpoofingBenchmarks/InTheWild`, `SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF`. Checkpoint chính thức:
`SpeechAntiSpoofingBenchmarks/Nes2Net` (bản rehost `nes2net_x_DF1.65.pth` từ Google Drive của tác giả).
"""

PLATFORM = {
 'kaggle': dict(
  title='Nes2Net — tái hiện trên Kaggle',
  howto=r"""
**Yêu cầu Kaggle:** Accelerator = GPU (T4 ×2 hoặc P100), **Internet = On**. Một phiên tối đa 12 giờ → chạy A+C trong
một phiên, B trong phiên riêng. Nếu B bị dừng giữa chừng, chạy lại notebook với cùng `/kaggle/working/runB`
(ví dụ add output của lần chạy trước làm input rồi copy vào) — huấn luyện sẽ **resume**.
""",
  setup=r"""
import os, subprocess, sys
REPO_DIR = '/tmp/nes2net-fake-voice-detection'   # ngoài /kaggle/working để không bị đóng gói vào output
""",
  paths=r"""
WORK = '/kaggle/working'            # được lưu lại sau phiên (tối đa ~20 GB)
TMP  = '/tmp/nes2net'               # đĩa tạm lớn, mất sau phiên
""",
  budget=r"""
# Part B (Kaggle-budget recipe; paper: lr=2.5e-7, 100 epochs)
B = dict(lr=1e-6, bs=12, epochs=100, algo=4, time_budget_h=8.3, dev='asv19:validation@6000', avg_k=3,
         keep_best_k=3, save_every_steps=500)
B_EVAL = False   # True: trung bình top-k checkpoint + đánh giá
# Part C
C = dict(train='asv19:train@6000', dev='asv19:validation@2000',
         evals=['asv19:test@4000', 'hfbench:InTheWild@4000'], layers=[24], seeds=[0, 1, 2], epochs=15,
         groups=['A0_baselines', 'A1_nest_ratio', 'A2_se_ratio', 'A3_pooling', 'A4_dilation_kernel', 'A5_connections'])
C_LAYER_STUDY = True   # thêm nghiên cứu chọn layer SSL (cache 4 layer, tập nhỏ hơn)
C_LAYER_N = dict(train=3000, dev=1000, itw=2000)
""",
  meta={'kaggle': {'accelerator': 'gpu', 'isInternetEnabled': True}}),
 'colab': dict(
  title='Nes2Net — tái hiện trên Google Colab',
  howto=r"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/""" + BLOB + r""")

**Chuẩn bị:** *Runtime → Change runtime type → GPU (T4 / L4 / A100)*. Ô Setup sẽ hỏi quyền **mount Google Drive** để
lưu checkpoint, score và bảng ablation — Colab có thể ngắt kết nối bất kỳ lúc nào (free: thường ≤ 12 giờ, ngắt
khi idle), nên phần **B lưu trạng thái định kỳ vào Drive và tự resume** khi bạn chạy lại notebook.

Lưu ý dung lượng: Drive free chỉ 15 GB. Mỗi checkpoint XLS-R ~1.3 GB, trạng thái resume (model + Adam) ~3.8 GB;
mặc định giữ 3 checkpoint tốt nhất → ~8 GB. Dữ liệu và cache đặc trưng nằm ở đĩa local `/content` (mất sau phiên,
tải lại tự động). Colab free chỉ có 2 CPU nên RawBoost (CPU) làm epoch chậm hơn Kaggle; Colab Pro + L4/A100 nhanh hơn nhiều.
Không dùng Drive? đặt `USE_DRIVE = False` (mọi thứ ở `/content`, mất khi runtime bị thu hồi).
""",
  setup=r"""
import os, subprocess, sys
USE_DRIVE = True
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB and USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
REPO_DIR = '/content/nes2net-fake-voice-detection' if IN_COLAB else os.path.abspath('nes2net-fake-voice-detection')
""",
  paths=r"""
if IN_COLAB:
    WORK = '/content/drive/MyDrive/nes2net' if USE_DRIVE else '/content/nes2net_work'   # bền vững (Drive)
    TMP  = '/content/nes2net_tmp'                                                       # đĩa local, nhanh
else:
    WORK, TMP = os.path.abspath('work'), os.path.abspath('tmp')
""",
  budget=r"""
# Part B (Colab: chia nhỏ theo phiên, resume từ Drive; paper: lr=2.5e-7, 100 epochs)
B = dict(lr=1e-6, bs=12, epochs=100, algo=4, time_budget_h=3.5, dev='asv19:validation@4000', avg_k=3,
         keep_best_k=3, save_every_steps=300)
B_EVAL = False   # True: trung bình top-k checkpoint + đánh giá
# Part C (nhỏ hơn bản Kaggle một chút: RAM 12.7 GB, đĩa ~80 GB trên Colab free)
C = dict(train='asv19:train@5000', dev='asv19:validation@2000',
         evals=['asv19:test@3000', 'hfbench:InTheWild@3000'], layers=[24], seeds=[0, 1, 2], epochs=15,
         groups=['A0_baselines', 'A1_nest_ratio', 'A2_se_ratio', 'A3_pooling', 'A4_dilation_kernel', 'A5_connections'])
C_LAYER_STUDY = True
C_LAYER_N = dict(train=2500, dev=1000, itw=2000)
""",
  meta={'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'T4'}}),
}


SMOKE_CELL = r"""
SMOKE = True
RUN_A = RUN_B = RUN_C = True
A_SETS = {
    'ITW@600 (4s)':              ('hfbench:InTheWild@600', False),
    'ASVspoof2019 LA eval@600':  ('asv19:test@600', False),
    'DF phase=eval (1 shard)@600': ('hfbench:ASVspoof2021_DF:1:eval@600', False),
}
A_ITW_FULL = False
B = dict(lr=1e-6, bs=8, epochs=2, algo=4, time_budget_h=0.25, dev='asv19:validation@200', avg_k=1,
         keep_best_k=1, save_every_steps=20)
B_EVAL = True
B_TRAIN = 'asv19:train@400'
B_EVAL_SETS = [('LA19 eval@300', 'asv19:test@300'), ('ITW@300', 'hfbench:InTheWild@300')]
C = dict(train='asv19:train@600', dev='asv19:validation@300', evals=['asv19:test@300', 'hfbench:InTheWild@300'],
         layers=[24], seeds=[0], epochs=3, groups=['A0_baselines', 'A5_connections'])
C_LAYER_STUDY = True
C_LAYER_N = dict(train=300, dev=150, itw=200)
print('SMOKE settings active')
"""


def build(platform: str, smoke: bool = False, out: str | None = None):
    P = PLATFORM[platform]
    C = []
    md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
    code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))
    md('# ' + P['title'] + '\n' + INTRO + P['howto'])
    code(P['setup'] + f"""
REPO_URL = "{REPO_URL}"
if not os.path.exists(REPO_DIR):
    subprocess.run(['git', 'clone', '-q', REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(['git', '-C', REPO_DIR, 'pull', '-q'], check=False)
sys.path.insert(0, REPO_DIR)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', os.path.join(REPO_DIR, 'requirements.txt')], check=True)
""")
    code(r"""
# ============================ Config ============================
RUN_A = True      # đánh giá checkpoint chính thức
RUN_B = False     # huấn luyện end-to-end -> nên chạy ở phiên riêng
RUN_C = True      # ablation trên đặc trưng đóng băng
""" + P['paths'] + r"""
os.makedirs(WORK, exist_ok=True); os.makedirs(TMP, exist_ok=True)
os.environ['HF_HOME'] = os.path.join(TMP, 'hf')     # cache dữ liệu/model lớn: KHÔNG để trên Drive
NW = max(1, min(4, os.cpu_count() or 2))             # DataLoader workers

# Part A
A_SETS = {
    'ITW (4s first window)':        ('hfbench:InTheWild', False),
    'ASVspoof2019 LA eval':         ('asv19:test', False),
    'ASVspoof2021 DF (phase=eval, 10/80 shards)': ('hfbench:ASVspoof2021_DF:10:eval', False),
    'ASVspoof2021 DF (all trials, 10/80 shards)': ('hfbench:ASVspoof2021_DF:10', False),
}
A_ITW_FULL = True   # thêm giao thức full-utterance cho ITW (paper dùng --test_protocol full cho ITW)
""" + P['budget'] + r"""
import torch, json, numpy as np, pandas as pd
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEV, torch.cuda.get_device_name(0) if DEV == 'cuda' else '', '| WORK =', WORK, '| workers =', NW)
""")
    if smoke:
        md("### ⚡ SMOKE RUN — tập con rất nhỏ, chỉ để kiểm tra toàn bộ pipeline chạy được (không dùng số liệu này)")
        code(SMOKE_CELL)
    exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_cells.py')).read(),
         {'md': md, 'code': code, 'PLATFORM_NAME': platform})
    nb = nbf.v4.new_notebook()
    nb['cells'] = C
    nb['metadata'] = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                      'language_info': {'name': 'python'}, **P['meta']}
    out = out or os.path.join(os.path.dirname(os.path.abspath(__file__)), f'nes2net_{platform}.ipynb')
    nbf.write(nb, out)
    print('wrote', out)


if __name__ == '__main__':
    for plat in PLATFORM:
        build(plat)
    build('kaggle', smoke=True, out=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kaggle',
                                                 'nes2net_kaggle_smoke.ipynb'))
