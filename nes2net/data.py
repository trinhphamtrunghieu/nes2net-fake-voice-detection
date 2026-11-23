"""Data utilities matching the official ``data_utils_SSL.py`` behaviour.

* Train/dev/eval (4 s protocol): take the FIRST 64,600 samples; shorter files are tiled.
* RawBoost is applied to the raw waveform before padding (train, and optionally dev).
* Label convention: 1 = bona fide, 0 = spoof (score = logit[:, 1]).

Audio sources supported: official ASVspoof folders + protocol files, In-the-Wild
(meta.csv), and HF Hub parquet datasets with an ``audio`` column (bytes, decode=False).
"""
from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable, Sequence

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from . import rawboost as RB

CUT = 64600  # ~4.04 s @ 16 kHz

RAWBOOST_DEFAULTS = SimpleNamespace(
    nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000, minCoeff=10, maxCoeff=100,
    minG=0, maxG=0, minBiasLinNonLin=5, maxBiasLinNonLin=20, N_f=5, P=10, g_sd=2,
    SNRmin=10, SNRmax=40)


@dataclass
class Item:
    key: str
    source: object          # path (str) or raw bytes
    label: int = -1         # 1 bona fide, 0 spoof, -1 unknown
    meta: dict | None = None


def pad(x: np.ndarray, max_len: int = CUT) -> np.ndarray:
    """Official pad: first max_len samples, tile-repeat if shorter."""
    if x.shape[0] >= max_len:
        return x[:max_len]
    reps = int(max_len / x.shape[0]) + 1
    return np.tile(x, reps)[:max_len]


class HFRef:
    """Lazy pointer to row ``idx`` of a memory-mapped HF/Arrow dataset (audio decode=False).
    Keeps RAM flat and is cheap to send to DataLoader workers."""
    __slots__ = ('ds', 'idx', 'col')

    def __init__(self, ds, idx, col='audio'):
        self.ds, self.idx, self.col = ds, idx, col

    def read(self):
        a = self.ds[self.idx][self.col]
        return a['bytes'] if a.get('bytes') else a['path']


def load_audio(source, sr: int = 16000) -> np.ndarray:
    if isinstance(source, HFRef):
        source = source.read()
    if isinstance(source, (bytes, bytearray)):
        x, fs = sf.read(io.BytesIO(source), dtype='float32')
    else:
        x, fs = sf.read(source, dtype='float32')
    if x.ndim > 1:
        x = x.mean(1)
    if fs != sr:
        import librosa
        x = librosa.resample(x, orig_sr=fs, target_sr=sr)
    return np.ascontiguousarray(x, dtype=np.float32)


def rawboost(x: np.ndarray, algo: int, sr: int = 16000, args=RAWBOOST_DEFAULTS) -> np.ndarray:
    """RawBoost algos 0-8 exactly as in the official repo (4 = 1+2+3 in series)."""
    a = args
    lnl = lambda y: RB.LnL_convolutive_noise(y, a.N_f, a.nBands, a.minF, a.maxF, a.minBW, a.maxBW,
                                             a.minCoeff, a.maxCoeff, a.minG, a.maxG,
                                             a.minBiasLinNonLin, a.maxBiasLinNonLin, sr)
    isd = lambda y: RB.ISD_additive_noise(y, a.P, a.g_sd)
    ssi = lambda y: RB.SSI_additive_noise(y, a.SNRmin, a.SNRmax, a.nBands, a.minF, a.maxF, a.minBW,
                                          a.maxBW, a.minCoeff, a.maxCoeff, a.minG, a.maxG, sr)
    if algo == 0:
        return x
    if algo == 1: return lnl(x)
    if algo == 2: return isd(x)
    if algo == 3: return ssi(x)
    if algo == 4: return ssi(isd(lnl(x)))
    if algo == 5: return isd(lnl(x))
    if algo == 6: return ssi(lnl(x))
    if algo == 7: return ssi(isd(x))
    if algo == 8: return RB.normWav(lnl(x) + isd(x), 0)
    raise ValueError(f'RawBoost algo {algo} not supported')


class WaveDataset(Dataset):
    """Returns (wave[T], label, key). full=True keeps the whole utterance (batch_size=1)."""

    def __init__(self, items: Sequence[Item], cut: int = CUT, algo: int = 0, full: bool = False):
        self.items, self.cut, self.algo, self.full = list(items), cut, algo, full

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        x = load_audio(it.source)
        if self.algo:
            x = rawboost(x, self.algo).astype(np.float32)
        if not self.full:
            x = pad(x, self.cut)
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), it.label, it.key


# ----------------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------------
def load_asvspoof_protocol(protocol: str, audio_dir: str, ext: str = '.flac',
                           phase: str | None = None) -> list[Item]:
    """Parses ASVspoof 2019 CM protocols (``SPK KEY - SYS label``) and 2021 keys
    (``trial_metadata.txt``: ``SPK KEY codec src SYS label trim phase ...``).
    ``phase`` filters 2021 keys to 'eval' | 'progress' | 'hidden_track' (official scoring uses 'eval')."""
    items = []
    with open(protocol) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            key = t[1] if len(t) > 1 else t[0]
            label = 1 if 'bonafide' in t else (0 if 'spoof' in t else -1)
            if phase is not None and phase not in t:
                continue
            meta = {'attack': next((x for x in t if x.startswith('A') and x[1:].isdigit()), '-'),
                    'raw': t}
            items.append(Item(key, os.path.join(audio_dir, key + ext), label, meta))
    return items


def load_in_the_wild(root: str) -> list[Item]:
    """In-the-Wild release: ``root/meta.csv`` with columns file,speaker,label (bona-fide|spoof)."""
    items = []
    with open(os.path.join(root, 'meta.csv')) as f:
        for row in csv.DictReader(f):
            lab = 1 if row['label'].strip().lower().startswith('bona') else 0
            items.append(Item(os.path.splitext(row['file'])[0], os.path.join(root, row['file']), lab,
                              {'speaker': row.get('speaker')}))
    return items


def hf_items(ds, label_fn, key_fn, meta_fn=None, audio_col='audio', keep=None) -> list[Item]:
    """Build lazy Items from an HF dataset. Metadata is read without touching audio."""
    from datasets import Audio
    ds = ds.cast_column(audio_col, Audio(decode=False))
    md = ds.remove_columns([audio_col]).to_list()
    items = []
    for j, r in enumerate(md):
        meta = meta_fn(r) if meta_fn else dict(r)
        if keep is not None and not keep(meta):
            continue
        items.append(Item(key_fn(r, j), HFRef(ds, j, audio_col), label_fn(r), meta))
    return items


def load_hf_parquet(repo_id: str, split: str, label_col: str = 'key', bonafide_value=0,
                    key_col: str | None = 'audio_file_name', audio_col: str = 'audio',
                    cache_dir: str | None = None) -> list[Item]:
    """Default arguments fit ``Bisher/ASVspoof_2019_LA`` (key: 0=bonafide, 1=spoof)."""
    from datasets import load_dataset
    ds = load_dataset(repo_id, split=split, cache_dir=cache_dir)

    def lab(r):
        v = r[label_col]
        return int(str(v).lower().startswith('bona')) if isinstance(v, str) else int(v == bonafide_value)
    return hf_items(ds, lab, lambda r, j: str(r[key_col]) if key_col in r else f'{split}_{j}',
                    audio_col=audio_col)


def balanced_subset(items: Sequence[Item], n: int, seed: int = 0, bona_frac: float | None = None):
    """Random subset of size n; optionally oversample bona fide to ``bona_frac``."""
    rng = np.random.default_rng(seed)
    items = list(items)
    if bona_frac is None:
        idx = rng.permutation(len(items))[:n]
        return [items[i] for i in idx]
    bona = [x for x in items if x.label == 1]
    spoof = [x for x in items if x.label == 0]
    nb = min(len(bona), int(n * bona_frac))
    return [bona[i] for i in rng.permutation(len(bona))[:nb]] + \
           [spoof[i] for i in rng.permutation(len(spoof))[:n - nb]]
