"""Training / scoring loops (mirrors official main.py, plus AMP, time budget, ckpt averaging)."""
from __future__ import annotations

import json
import math
import os
import time
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import Item, WaveDataset
from .metrics import compute_eer, compute_min_dcf


def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weighted_ce(device):
    # official: weight [0.1, 0.9] for [spoof, bona fide] (class imbalance ~ 1:9)
    return nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9], device=device))


def _loader(items, bs, algo=0, shuffle=False, workers=4, full=False, drop_last=False):
    ds = WaveDataset(items, algo=algo, full=full)
    return DataLoader(ds, batch_size=1 if full else bs, shuffle=shuffle, num_workers=workers,
                      drop_last=drop_last, pin_memory=torch.cuda.is_available(),
                      persistent_workers=workers > 0)


@torch.no_grad()
def score(model, items: Sequence[Item], bs=32, device='cuda', full=False, workers=4,
          amp=True, desc='score'):
    """Returns dict key -> score (logit of bona fide class)."""
    from tqdm.auto import tqdm
    model.eval()
    out = {}
    for x, y, k in tqdm(_loader(items, bs, full=full, workers=workers), desc=desc, leave=False):
        x = x.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp and device.startswith('cuda')):
            s = model(x)[:, 1].float().cpu().numpy()
        out.update(zip(k, s.tolist()))
    return out


def evaluate_scores(scores: dict, items: Sequence[Item], by_attack=True) -> dict:
    lab = {it.key: it.label for it in items}
    keys = [k for k in scores if lab.get(k, -1) >= 0]
    s = np.array([scores[k] for k in keys]); y = np.array([lab[k] for k in keys])
    res = {'n': int(len(keys)), 'eer': 100 * compute_eer(s[y == 1], s[y == 0])[0],
           'minDCF': compute_min_dcf(s[y == 1], s[y == 0])}
    if by_attack:
        att = {it.key: (it.meta or {}).get('attack', '-') for it in items}
        bona = s[y == 1]
        per = {}
        for a in sorted({att[k] for k, yy in zip(keys, y) if yy == 0}):
            if a in ('-', None):
                continue
            sp = np.array([scores[k] for k, yy in zip(keys, y) if yy == 0 and att[k] == a])
            if len(sp):
                per[a] = 100 * compute_eer(bona, sp)[0]
        if per:
            res['per_attack_eer'] = per
    return res


def write_scores(scores: dict, path: str):
    with open(path, 'w') as f:
        for k, v in scores.items():
            f.write(f'{k} {v}\n')


@torch.no_grad()
def val_loss(model, loader, crit, device, amp=True):
    model.eval(); tot, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device), torch.as_tensor(y).long().to(device)
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp and device.startswith('cuda')):
            out = model(x)
        tot += crit(out.float(), y).item() * x.size(0); n += x.size(0)
    return tot / max(n, 1)


def train(model, train_items, dev_items, out_dir, epochs=100, lr=1e-6, wd=1e-4, bs=12,
          algo=4, val_aug=True, device='cuda', amp=True, accum=1, workers=4,
          time_budget_h: float | None = None, log=print):
    """Official recipe: Adam(lr, wd=1e-4), weighted CE, RawBoost on train; keep a checkpoint
    whenever val loss (w/ or w/o aug) improves. Returns history list."""
    os.makedirs(out_dir, exist_ok=True)
    model.to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    scaler = torch.amp.GradScaler('cuda', enabled=amp and device.startswith('cuda'))
    crit = weighted_ce(device)
    tr = _loader(train_items, bs, algo=algo, shuffle=True, workers=workers, drop_last=True)
    dv0 = _loader(dev_items, bs, algo=0, workers=workers)
    dv1 = _loader(dev_items, bs, algo=algo, workers=workers) if val_aug and algo else None
    best0 = best1 = math.inf
    hist, t0 = [], time.time()
    for ep in range(1, epochs + 1):
        model.train(); run, n = 0.0, 0; te = time.time()
        opt.zero_grad(set_to_none=True)
        for step, (x, y, _) in enumerate(tr):
            x, y = x.to(device, non_blocking=True), torch.as_tensor(y).long().to(device)
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp and device.startswith('cuda')):
                out = model(x)
            loss = crit(out.float(), y) / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            run += loss.item() * accum * x.size(0); n += x.size(0)
            if time_budget_h and (time.time() - t0) / 3600 > time_budget_h:
                break
        l0 = val_loss(model, dv0, crit, device, amp)
        l1 = val_loss(model, dv1, crit, device, amp) if dv1 is not None else float('nan')
        save = False
        if l0 <= best0: best0, save = l0, True
        if dv1 is not None and l1 <= best1: best1, save = l1, True
        if save:
            torch.save(model.state_dict(), os.path.join(out_dir, f'epoch_{ep}.pth'))
        h = dict(epoch=ep, train_loss=run / max(n, 1), val_loss_wo_aug=l0, val_loss_w_aug=l1,
                 saved=save, sec=time.time() - te)
        hist.append(h); log(json.dumps(h))
        with open(os.path.join(out_dir, 'history.json'), 'w') as f:
            json.dump(hist, f, indent=1)
        if time_budget_h and (time.time() - t0) / 3600 > time_budget_h:
            log(f'time budget reached after epoch {ep}'); break
    return hist


def average_checkpoints(model, paths: Sequence[str], out_path: str | None = None):
    """Uniform weight averaging (official --num_average_model). BN running stats are averaged too."""
    sd = None
    for p in paths:
        s = torch.load(p, map_location='cpu')
        if sd is None:
            sd = {k: v.clone().double() if v.is_floating_point() else v.clone() for k, v in s.items()}
        else:
            for k in sd:
                if sd[k].is_floating_point():
                    sd[k] += s[k].double()
    for k in sd:
        if sd[k].is_floating_point():
            sd[k] = (sd[k] / len(paths)).float()
    model.load_state_dict(sd)
    if out_path:
        torch.save(sd, out_path)
    return model


def best_epochs(hist, k=5, key='val_loss_wo_aug'):
    saved = [h for h in hist if h['saved'] and not math.isnan(h[key])]
    return [h['epoch'] for h in sorted(saved, key=lambda h: h[key])[:k]]
