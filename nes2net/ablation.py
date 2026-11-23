"""Fast back-end ablations on *cached, frozen* SSL features.

Full end-to-end fine-tuning (paper setting) costs ~30-40 min/epoch for XLS-R on a Kaggle T4,
so a grid of ablations x seeds is infeasible there.  Here the SSL model is frozen, features
of a fixed 4 s window are cached once (fp16 memmap), and each back-end trains in minutes.
Absolute EERs are therefore higher than the paper's (frozen SSL, subset of data); only
*relative* comparisons between back-end variants are meaningful, and every config is run
with several seeds so the spread is visible.
"""
from __future__ import annotations

import json
import os
import time
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .backend import build_backend, count_macs, count_params
from .data import Item, WaveDataset
from .metrics import compute_eer


# ------------------------------------------------------------------ feature cache
@torch.no_grad()
def extract_features(ssl_hf_model, items: Sequence[Item], out_dir: str, layers=(24,), bs=32,
                     device='cuda', workers=4, amp=True):
    """Writes ``out_dir/L{layer}.npy`` (N, C, T) fp16 memmaps + ``labels.npy`` + ``keys.json``.
    ``ssl_hf_model`` is a raw HF Wav2Vec2Model/WavLMModel."""
    from tqdm.auto import tqdm
    os.makedirs(out_dir, exist_ok=True)
    done = os.path.join(out_dir, 'DONE')
    if os.path.exists(done):
        return out_dir
    ssl_hf_model.eval().to(device)
    dl = DataLoader(WaveDataset(items), batch_size=bs, num_workers=workers)
    N = len(items)
    mm, keys, labels, i = {}, [], [], 0
    for x, y, k in tqdm(dl, desc=f'extract {os.path.basename(out_dir)}'):
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp and device.startswith('cuda')):
            hs = ssl_hf_model(x.to(device), output_hidden_states=True).hidden_states
        for L in layers:
            h = hs[L].transpose(1, 2).half().cpu().numpy()        # B C T
            if L not in mm:
                mm[L] = np.lib.format.open_memmap(os.path.join(out_dir, f'L{L}.npy'), mode='w+',
                                                  dtype=np.float16, shape=(N,) + h.shape[1:])
            mm[L][i:i + len(h)] = h
        keys += list(k); labels += [int(v) for v in y]; i += len(x)
    for m in mm.values():
        m.flush()
    np.save(os.path.join(out_dir, 'labels.npy'), np.array(labels))
    json.dump({'keys': keys, 'layers': list(layers)}, open(os.path.join(out_dir, 'keys.json'), 'w'))
    open(done, 'w').close()
    return out_dir


class FeatDataset(Dataset):
    def __init__(self, cache_dir: str, layers: Sequence[int] = (24,), in_ram=False):
        self.y = np.load(os.path.join(cache_dir, 'labels.npy'))
        self.keys = json.load(open(os.path.join(cache_dir, 'keys.json')))['keys']
        self.X = [np.load(os.path.join(cache_dir, f'L{L}.npy'), mmap_mode=None if in_ram else 'r')
                  for L in layers]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = np.stack([X[i] for X in self.X], 0).astype(np.float32)   # L C T
        return torch.from_numpy(x), int(self.y[i])


class LayerMix(nn.Module):
    """Softmax-weighted sum over cached layers (L C T -> C T), followed by a back-end."""

    def __init__(self, n_layers: int, backend: nn.Module):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_layers))
        self.backend = backend

    def forward(self, x):
        if x.ndim == 4:
            x = (torch.softmax(self.w, 0).view(1, -1, 1, 1) * x).sum(1)
        return self.backend(x)


# ------------------------------------------------------------------ train / eval
@torch.no_grad()
def _scores(model, dl, device):
    model.eval(); S, Y = [], []
    for x, y in dl:
        S.append(model(x.to(device))[:, 1].float().cpu()); Y.append(y)
    return torch.cat(S).numpy(), torch.cat(Y).numpy()


def _eer(s, y):
    return 100 * compute_eer(s[y == 1], s[y == 0])[0]


def train_backend(cfg: dict, train_dir, dev_dir, eval_dirs: dict, layers=(24,), seed=0,
                  epochs=15, lr=1e-3, wd=1e-4, bs=64, device='cuda', workers=2, in_ram=False):
    """cfg: {'backend': 'nes2net_x', ...backend kwargs}. Model selection by dev EER."""
    torch.manual_seed(seed); np.random.seed(seed)
    cfg = dict(cfg); name = cfg.pop('backend')
    be = build_backend(name, input_channel=1024, n_output_logits=2, **cfg)
    model = LayerMix(len(layers), be).to(device)
    mk = lambda d, sh: DataLoader(FeatDataset(d, layers, in_ram), batch_size=bs, shuffle=sh,
                                  num_workers=workers, drop_last=sh)
    tr, dv = mk(train_dir, True), mk(dev_dir, False)
    ev = {k: mk(v, False) for k, v in eval_dirs.items()}
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9], device=device))
    best, best_sd, t0 = 1e9, None, time.time()
    for ep in range(epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        e = _eer(*_scores(model, dv, device))
        if e < best:
            best, best_sd = e, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_sd)
    res = {'backend': name, **cfg, 'seed': seed, 'dev_eer': best,
           'params_k': count_params(be) / 1e3, 'mmacs': count_macs(be), 'train_min': (time.time() - t0) / 60}
    for k, d in ev.items():
        res[f'{k}_eer'] = _eer(*_scores(model, d, device))
    return res


# The grid used in docs/ablation_and_gaps.md. Keys = ablation group.
ABLATIONS = {
    'A0_baselines': [
        {'backend': 'mean_linear'},
        {'backend': 'dr_res2net', 'D': 128, 'scale': 4},
        {'backend': 'nes2net'},
        {'backend': 'nes2net_x'},
    ],
    'A1_nest_ratio': [{'backend': 'nes2net_x', 'Nes_ratio': r} for r in
                      ([4, 4], [4, 8], [8, 4], [8, 8], [8, 16], [16, 4], [16, 8], [32, 4])],
    'A2_se_ratio': [{'backend': 'nes2net_x', 'SE_ratio': [r]} for r in (1, 4, 8, 16)] +
                   [{'backend': 'nes2net_x', 'use_se': False}],
    'A3_pooling': [{'backend': 'nes2net_x', 'pool_func': p} for p in ('mean', 'ASTP')],
    'A4_dilation_kernel': [{'backend': 'nes2net_x', 'dilation': d} for d in (1, 2, 3)] +
                          [{'backend': 'nes2net_x', 'kernel_size': 5, 'dilation': 1}],
    'A5_connections': [
        {'backend': 'nes2net_x', 'outer_hier': False},
        {'backend': 'nes2net_x', 'inner_hier': False},
        {'backend': 'nes2net_x', 'outer_hier': False, 'inner_hier': False},
    ],
}


def run_grid(groups: Sequence[str], train_dir, dev_dir, eval_dirs, seeds=(0, 1, 2), out_csv=None,
             layers=(24,), **kw):
    import pandas as pd
    rows = []
    if out_csv and os.path.exists(out_csv):
        rows = pd.read_csv(out_csv).to_dict('records')
    done = {(r['group'], r['cfg'], r['seed']) for r in rows}
    for g in groups:
        for cfg in ABLATIONS[g]:
            for s in seeds:
                tag = json.dumps(cfg, sort_keys=True)
                if (g, tag, s) in done:
                    continue
                r = train_backend(cfg, train_dir, dev_dir, eval_dirs, layers=layers, seed=s, **kw)
                r.update(group=g, cfg=tag); rows.append(r)
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
                if out_csv:
                    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return pd.DataFrame(rows)


def summarize(df):
    """mean ± std over seeds per config."""
    cols = [c for c in df.columns if c.endswith('_eer')]
    g = df.groupby(['group', 'cfg'])
    out = g[cols].agg(['mean', 'std']).round(2)
    out[('params_k', '')] = g['params_k'].first().round(1)
    out[('mmacs', '')] = g['mmacs'].first().round(1)
    return out
