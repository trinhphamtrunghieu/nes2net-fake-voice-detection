"""Text-matched TTS probe sets for zero-shot cross-lingual checks (e.g. Vietnamese).

bona fide : FLEURS read speech for one language (``google/fleurs``, parquet-data/<lang>/<split>)
spoof     : the SAME sentences synthesised by an open TTS (default MMS-TTS / VITS for that language)

Pairing by sentence removes the "content" shortcut. Two cues can remain and are reported
separately instead of hidden:
  * level: FLEURS recordings and TTS output have different loudness -> every item is also
    available RMS-normalised to the same level (``norm=True``)
  * channel: FLEURS has varied microphones/rooms, TTS is clean. An English control probe built
    the same way (FLEURS en_us + MMS-TTS eng) tells how much of the error is the *language*
    shift rather than the probe protocol itself.
This is a quick probe, not a benchmark: one TTS system, read speech, no codec/VC attacks.
"""
from __future__ import annotations

import io
import json
import os
from typing import Callable, Sequence

import numpy as np
import soundfile as sf

from .data import Item

TARGET_RMS = 0.05   # ~ -26 dBFS


def _rms_norm(x: np.ndarray, target=TARGET_RMS) -> np.ndarray:
    r = float(np.sqrt(np.mean(x ** 2)) + 1e-9)
    y = x * (target / r)
    peak = float(np.max(np.abs(y)) + 1e-9)
    return (y / peak * 0.99 if peak > 0.99 else y).astype(np.float32)


def _wav_bytes(x: np.ndarray, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x, sr, format='FLAC')
    return buf.getvalue()


def load_fleurs(lang: str = 'vi_vn', split: str = 'test', limit: int | None = None,
                local_dir: str | None = None) -> list[dict]:
    """Rows of FLEURS (one parquet per split): id, raw_transcription, gender, audio(16 kHz)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    p = hf_hub_download('google/fleurs', f'parquet-data/{lang}/{split}-00000-of-00001.parquet',
                        repo_type='dataset', local_dir=local_dir)
    rows, seen = [], set()
    for r in pq.read_table(p).to_pylist():
        sent = (r.get('raw_transcription') or r.get('transcription') or '').strip()
        if not sent or sent in seen:          # FLEURS has several speakers per sentence: keep one
            continue
        seen.add(sent)
        a = r['audio']
        x, fs = sf.read(io.BytesIO(a['bytes']), dtype='float32') if a.get('bytes') else sf.read(a['path'], dtype='float32')
        if x.ndim > 1:
            x = x.mean(1)
        if fs != 16000:
            import librosa
            x = librosa.resample(x, orig_sr=fs, target_sr=16000)
        g = r.get('gender')
        g = {0: 'male', 1: 'female'}.get(g, g) if isinstance(g, int) else (str(g).lower() if g else None)
        rows.append({'id': str(r.get('id')), 'text': sent, 'gender': g, 'audio': x})
        if limit and len(rows) >= limit:
            break
    return rows


def mms_tts(model_id: str = 'facebook/mms-tts-vie', device: str = 'cuda', seed: int = 0) -> Callable[[str], np.ndarray]:
    """Return text -> 16 kHz waveform using an MMS (VITS) checkpoint from transformers."""
    import torch
    from transformers import AutoTokenizer, VitsModel
    tok = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id).to(device).eval()
    sr = model.config.sampling_rate
    g = torch.Generator(device='cpu').manual_seed(seed)

    @torch.no_grad()
    def synth(text: str) -> np.ndarray:
        torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=g)))
        inp = tok(text, return_tensors='pt').to(device)
        if inp['input_ids'].shape[-1] < 2:          # nothing the MMS vocabulary can pronounce
            return None
        y = model(**inp).waveform[0].float().cpu().numpy()
        if sr != 16000:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        return y.astype(np.float32)
    return synth


def build_probe(lang: str, tts_model: str, cache_dir: str, split: str = 'test', limit: int | None = None,
                device: str = 'cuda', synth: Callable[[str], np.ndarray] | None = None,
                rows: Sequence[dict] | None = None, log=print) -> dict[str, list[Item]]:
    """Returns {'raw': items, 'norm': items}; spoof audio is cached as FLAC in cache_dir."""
    os.makedirs(cache_dir, exist_ok=True)
    rows = rows if rows is not None else load_fleurs(lang, split, limit, os.path.join(cache_dir, '_fleurs'))
    synth = synth or mms_tts(tts_model, device)
    meta_p = os.path.join(cache_dir, 'meta.json')
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    raw, norm = [], []
    for i, r in enumerate(rows):
        f = os.path.join(cache_dir, f'tts_{r["id"]}.flac')
        if not os.path.exists(f):
            y = synth(r['text'])
            if y is None or len(y) < 1600:           # skip sentences the TTS cannot render (< 0.1 s)
                continue
            sf.write(f, y, 16000, format='FLAC')
            meta[r['id']] = {'text': r['text'], 'tts': tts_model}
        y, _ = sf.read(f, dtype='float32')
        x = r['audio']
        m = {'lang': lang, 'gender': r.get('gender'), 'text_id': r['id'], 'attack': tts_model.split('/')[-1]}
        raw += [Item(f'bona_{r["id"]}', _wav_bytes(x), 1, dict(m, attack='-')),
                Item(f'tts_{r["id"]}', _wav_bytes(y), 0, m)]
        norm += [Item(f'bona_{r["id"]}', _wav_bytes(_rms_norm(x)), 1, dict(m, attack='-')),
                 Item(f'tts_{r["id"]}', _wav_bytes(_rms_norm(y)), 0, m)]
        if (i + 1) % 100 == 0:
            log(f'{lang}: {i + 1}/{len(rows)} sentences ready')
    json.dump(meta, open(meta_p, 'w'), ensure_ascii=False, indent=0)
    dur = lambda items, lab: float(np.mean([sf.info(io.BytesIO(it.source)).duration for it in items if it.label == lab]))
    log(f'{lang}: {len(rows)} pairs | mean dur bona {dur(raw, 1):.1f}s / tts {dur(raw, 0):.1f}s')
    return {'raw': raw, 'norm': norm}
