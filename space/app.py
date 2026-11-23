"""Nes2Net-X speech anti-spoofing demo (HF Space, ZeroGPU).

Model: official wav2vec 2.0 XLS-R 300M + Nes2Net-X checkpoint (ASVspoof2019 LA + RawBoost),
       rehosted at SpeechAntiSpoofingBenchmarks/Nes2Net (nes2net_x_DF1.65.pth).
Loaded without fairseq: fairseq keys are converted to HF Wav2Vec2Model (see nes2net/model.py).
"""
try:                       # ZeroGPU: `spaces` must be imported before torch
    import spaces
except ImportError:        # local run without the HF runtime
    class spaces:          # noqa: N801
        @staticmethod
        def GPU(*a, **k):
            return (lambda f: f) if not (a and callable(a[0])) else a[0]

import os
import time

import gradio as gr
import numpy as np
import pandas as pd
import torch

from nes2net.data import CUT, load_audio, pad
from nes2net.model import load_official_checkpoint

CKPT_REPO = os.environ.get('CKPT_REPO', 'SpeechAntiSpoofingBenchmarks/Nes2Net')
CKPT_FILE = os.environ.get('CKPT_FILE', 'nes2net_x_DF1.65.pth')
MAX_SEC = 60
DEVICE = 'cuda' if (torch.cuda.is_available() or os.environ.get('SPACES_ZERO_GPU')) else 'cpu'


def _build_model():
    if os.environ.get('DEMO_FAKE'):          # UI test without the 1.3 GB checkpoint
        from transformers import Wav2Vec2Config, Wav2Vec2Model
        from nes2net.frontend import XLSR_300M_CONFIG
        from nes2net.model import SSLNes2Net
        m = SSLNes2Net(pretrained_ssl=False)
        m.ssl_model.model = Wav2Vec2Model(Wav2Vec2Config(**dict(XLSR_300M_CONFIG, num_hidden_layers=1)))
        return m.eval()
    from huggingface_hub import hf_hub_download
    return load_official_checkpoint(hf_hub_download(CKPT_REPO, CKPT_FILE)).eval()


# ZeroGPU rule: load at module scope and move to "cuda" eagerly
MODEL = _build_model().to(DEVICE)


def _windows(x: np.ndarray, hop: int):
    if len(x) <= CUT:
        return [pad(x, CUT)], [0.0]
    starts = list(range(0, len(x) - CUT + 1, hop))
    if starts[-1] + CUT < len(x):
        starts.append(len(x) - CUT)
    return [x[s:s + CUT] for s in starts], [s / 16000 for s in starts]


def _infer(batch: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(batch).float().to(DEVICE)
        return torch.cat([MODEL(x[i:i + 16]) for i in range(0, len(x), 16)]).float().cpu().numpy()


@spaces.GPU(duration=30)
def detect_spoof(audio_path: str, mode: str = 'Cửa sổ trượt (toàn câu)', threshold: float = 0.0):
    """Detect whether a speech recording is bona fide (real human) or spoofed (TTS / voice conversion).

    Args:
        audio_path: path or URL to an audio file (wav/flac/mp3), any sample rate; first 60 s are used.
        mode: 'Paper (4 s đầu)' scores only the first 4.04 s window (official protocol);
              'Cửa sổ trượt (toàn câu)' averages 4.04 s windows with 2 s hop over the whole clip.
        threshold: decision threshold on the mean log-odds (bona fide minus spoof logit); > threshold => bona fide.

    Returns:
        A markdown verdict, and a table of P(bona fide) per window over time.
    """
    if audio_path is None:
        return 'Hãy tải lên hoặc ghi âm một đoạn audio.', None
    x = load_audio(audio_path)
    dur = len(x) / 16000
    if dur < 0.3:
        return 'Audio quá ngắn (< 0.3 s).', None
    x = x[: 16000 * MAX_SEC]
    t0 = time.perf_counter()
    if mode.startswith('Paper'):
        wins, starts = [pad(x, CUT)], [0.0]
    else:
        wins, starts = _windows(x, hop=CUT // 2)
    logits = _infer(np.stack(wins))
    score = logits[:, 1] - logits[:, 0]
    p_bona = 1 / (1 + np.exp(-score))
    s = float(score.mean())
    verdict = 'BONA FIDE (giọng thật)' if s >= threshold else 'SPOOF (giọng giả / tổng hợp)'
    md = (f"### {verdict}\n"
          f"- Điểm log-odds trung bình: **{s:+.3f}** (ngưỡng {threshold:+.2f}; > ngưỡng ⇒ thật)\n"
          f"- P(bona fide) trung bình (softmax, *chưa hiệu chỉnh*): **{p_bona.mean():.3f}**\n"
          f"- Thời lượng: {dur:.1f} s · {len(wins)} cửa sổ 4.04 s · suy luận {time.perf_counter() - t0:.2f} s\n"
          f"- Raw score kiểu ASVspoof (logit lớp bona fide, cửa sổ đầu): {float(logits[0, 1]):+.3f}")
    return md, pd.DataFrame({'t (s)': starts, 'P(bona fide)': p_bona.round(4).tolist()})


DESC = """
**Nes2Net-X** ([Liu et al., IEEE T-IFS 2025](https://arxiv.org/abs/2504.05657)): back-end *Nested Res2Net* chỉ
khoảng 511k tham số, chạy thẳng trên đặc trưng 1024 chiều của wav2vec 2.0 XLS-R, không cần lớp giảm chiều.
Checkpoint chính thức được huấn luyện trên **ASVspoof 2019 LA** (+ RawBoost).
Code tái hiện: [GitHub](https://github.com/trinhphamtrunghieu/nes2net-fake-voice-detection).

⚠️ Đây là demo nghiên cứu. Với giọng tổng hợp hiện đại, nhiễu, nhạc nền hoặc ngôn ngữ khác tiếng Anh, độ chính
xác giảm mạnh (xem tab *Giới thiệu*). Không dùng làm bằng chứng pháp lý.
"""

ABOUT = """
### Kiến trúc
`waveform 16 kHz → XLS-R 300M (24 layer, 1024-d, fine-tune) → Nes2Net-X → mean pooling → FC(2)`

Nes2Net chia 1024 kênh thành 8 nhóm (tầng ngoài). Nhóm i nhận `x_i + y_{i-1}` rồi đi qua một khối Res2Net bên
trong (scale 8, có SE). Bản -X thay phép cộng ở tầng trong bằng phép ghép cộng tổng có trọng số học được.

### Kết quả của checkpoint này (EER %, càng thấp càng tốt)
| Tập | Paper | Đánh giá độc lập (Speech Anti-Spoofing Arena) |
|---|---|---|
| ASVspoof 2021 DF | 1.65 (phase *eval*) | 3.61 (toàn bộ 611k trial) |
| ASVspoof 2021 LA | 1.73 (phase *eval*) | 6.14 (toàn bộ) |
| In-the-Wild | 5.52 (checkpoint khác, cả câu) | 8.48 (4 s đầu) |

Chênh lệch chủ yếu đến từ **giao thức đánh giá**: tập con phase=eval hay toàn bộ trial (gồm cả hidden track),
và cắt 4 s đầu hay dùng cả câu.

### Chế độ
* **Paper (4 s đầu):** đúng giao thức chính thức, lấy 64,600 mẫu đầu, lặp lại nếu audio ngắn hơn.
* **Cửa sổ trượt:** chia câu thành các cửa sổ 4 s (bước 2 s) rồi lấy trung bình; giúp quan sát *partial spoof*.
"""

CSS = """
#col-container { max-width: 1100px; margin: 0 auto; }
.dark .gradio-container { color: var(--body-text-color); }
"""

with gr.Blocks(title='Nes2Net-X anti-spoofing') as demo:
    with gr.Column(elem_id='col-container'):
        gr.Markdown('# 🎙️ Nes2Net-X: phát hiện giọng nói giả mạo')
        gr.Markdown(DESC)
        with gr.Tab('Demo'):
            with gr.Row():
                with gr.Column():
                    inp = gr.Audio(sources=['upload', 'microphone'], type='filepath', label='Audio (wav/flac/mp3)')
                    mode = gr.Radio(['Paper (4 s đầu)', 'Cửa sổ trượt (toàn câu)'],
                                    value='Cửa sổ trượt (toàn câu)', label='Giao thức')
                    thr = gr.Slider(-5, 5, value=0.0, step=0.1, label='Ngưỡng log-odds')
                    btn = gr.Button('Phân tích', variant='primary')
                with gr.Column():
                    out = gr.Markdown()
                    plot = gr.LinePlot(x='t (s)', y='P(bona fide)', y_lim=[0, 1], label='P(bona fide) theo thời gian')
            btn.click(detect_spoof, [inp, mode, thr], [out, plot], api_name='detect_spoof')
        with gr.Tab('Giới thiệu'):
            gr.Markdown(ABOUT)

if __name__ == '__main__':
    try:
        import mcp  # noqa: F401  (gradio[mcp] extra)
        _mcp = True
    except ImportError:
        _mcp = False
    demo.queue(max_size=16).launch(mcp_server=_mcp, theme=gr.themes.Citrus(), css=CSS)
