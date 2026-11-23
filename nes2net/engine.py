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


def _prune(out_dir, hist, keep_k):
    """Keep only the keep_k best saved epochs (by val loss w/o aug, then w/ aug) on disk."""
    keep = set(best_epochs(hist, keep_k, 'val_loss_wo_aug')) | set(best_epochs(hist, keep_k, 'val_loss_w_aug'))
    for h in hist:
        f = os.path.join(out_dir, f'epoch_{h["epoch"]}.pth')
        if h['saved'] and h['epoch'] not in keep and os.path.exists(f):
            os.remove(f)


def train(model, train_items, dev_items, out_dir, epochs=100, lr=1e-6, wd=1e-4, bs=12,
          algo=4, val_aug=True, device='cuda', amp=True, accum=1, workers=4,
          time_budget_h: float | None = None, log=print, resume: bool = True,
          keep_best_k: int | None = None, save_every_steps: int | None = None):
    """Official recipe: Adam(lr, wd=1e-4), weighted CE, RawBoost on train; keep a checkpoint
    whenever val loss (w/ or w/o aug) improves. Returns history list.

    Extras for preemptible notebooks (Colab/Kaggle):
      resume=True          continue from ``out_dir/last_state.pth`` (model + Adam + scaler + history)
      save_every_steps=N   also refresh last_state.pth every N steps inside an epoch; on resume the
                           interrupted epoch runs only its remaining steps (with a new shuffle)
      keep_best_k=K        delete epoch checkpoints outside the K best (saves disk / Drive quota)
    """
    os.makedirs(out_dir, exist_ok=True)
    model.to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    scaler = torch.amp.GradScaler('cuda', enabled=amp and device.startswith('cuda'))
    crit = weighted_ce(device)
    tr = _loader(train_items, bs, algo=algo, shuffle=True, workers=workers, drop_last=True)
    dv0 = _loader(dev_items, bs, algo=0, workers=workers)
    dv1 = _loader(dev_items, bs, algo=algo, workers=workers) if val_aug and algo else None
    best0 = best1 = math.inf
    hist, start_ep, start_step, carry = [], 1, 0, (0.0, 0)
    state_path = os.path.join(out_dir, 'last_state.pth')

    def save_state(ep, step, run, n):
        tmp = state_path + '.tmp'
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(),
                    'epoch': ep, 'step': step, 'run': run, 'n': n, 'best0': best0, 'best1': best1,
                    'hist': hist, 'rng': torch.get_rng_state()}, tmp)
        os.replace(tmp, state_path)            # atomic: a disconnect never leaves a broken file

    if resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location='cpu', weights_only=False)
        model.load_state_dict(st['model']); opt.load_state_dict(st['opt']); scaler.load_state_dict(st['scaler'])
        best0, best1, hist = st['best0'], st['best1'], st['hist']
        torch.set_rng_state(st['rng'])
        if st['step'] == 0:                      # saved at end of epoch
            start_ep = st['epoch'] + 1
        else:                                    # saved mid-epoch
            start_ep, start_step, carry = st['epoch'], st['step'], (st['run'], st['n'])
        log(f'resumed from {state_path}: epoch {start_ep}, step {start_step}')

    t0 = time.time()
    out_of_time = lambda: bool(time_budget_h) and (time.time() - t0) / 3600 > time_budget_h
    for ep in range(start_ep, epochs + 1):
        model.train(); te = time.time()
        run, n = carry if ep == start_ep else (0.0, 0)
        skip = start_step if ep == start_ep else 0
        opt.zero_grad(set_to_none=True)
        stopped = False
        for step, (x, y, _) in enumerate(tr, start=skip):
            if step >= len(tr):
                break
            x, y = x.to(device, non_blocking=True), torch.as_tensor(y).long().to(device)
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp and device.startswith('cuda')):
                out = model(x)
            loss = crit(out.float(), y) / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            run += loss.item() * accum * x.size(0); n += x.size(0)
            if save_every_steps and (step + 1) % save_every_steps == 0 and step + 1 < len(tr):
                save_state(ep, step + 1, run, n)
            if out_of_time():
                save_state(ep, step + 1, run, n); stopped = True
                break
        if stopped and n and len(tr) - (step + 1) > 0:
            log(f'time budget reached mid-epoch {ep} (step {step + 1}/{len(tr)}); state saved, rerun to resume')
            return hist
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
        if keep_best_k:
            _prune(out_dir, hist, keep_best_k)
        save_state(ep, 0, 0.0, 0)
        if out_of_time():
            log(f'time budget reached after epoch {ep}; rerun to resume'); break
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
