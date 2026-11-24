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
from .logutil import RunLogger, get_logger, gpu_mem_gb
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
    out, t0 = {}, time.time()
    for x, y, k in tqdm(_loader(items, bs, full=full, workers=workers), desc=desc, leave=False):
        x = x.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp and device.startswith('cuda')):
            s = model(x)[:, 1].float().cpu().numpy()
        out.update(zip(k, s.tolist()))
    dt = time.time() - t0
    get_logger('nes2net.score').info('%s: %d utts in %.1f min (%.1f utt/s)%s', desc, len(out), dt / 60,
                                     len(out) / max(dt, 1e-9), ' [full utt]' if full else '')
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
def val_loss(model, loader, crit, device, amp=True, return_eer=False):
    """Weighted-CE loss on a loader; with return_eer=True also the EER (%) of logit[:, 1]."""
    model.eval(); tot, n = 0.0, 0
    S, Y = [], []
    for x, y, _ in loader:
        x, y = x.to(device), torch.as_tensor(y).long().to(device)
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp and device.startswith('cuda')):
            out = model(x)
        tot += crit(out.float(), y).item() * x.size(0); n += x.size(0)
        if return_eer:
            S.append(out[:, 1].float().cpu()); Y.append(y.cpu())
    loss = tot / max(n, 1)
    if not return_eer:
        return loss
    s, yy = torch.cat(S).numpy(), torch.cat(Y).numpy()
    eer = 100 * compute_eer(s[yy == 1], s[yy == 0])[0] if (yy == 1).any() and (yy == 0).any() else float('nan')
    return loss, eer


def _prune(out_dir, hist, keep_k):
    """Keep only the keep_k best saved epochs (by val loss w/o aug, then w/ aug) on disk."""
    keep = set(best_epochs(hist, keep_k, 'val_loss_wo_aug')) | set(best_epochs(hist, keep_k, 'val_loss_w_aug'))
    for h in hist:
        f = os.path.join(out_dir, f'epoch_{h["epoch"]}.pth')
        if h['saved'] and h['epoch'] not in keep and os.path.exists(f):
            os.remove(f)


def train(model, train_items, dev_items, out_dir, epochs=100, lr=1e-6, wd=1e-4, bs=12,
          algo=4, val_aug=True, device='cuda', amp=True, accum=1, workers=4,
          time_budget_h: float | None = None, log=None, resume: bool = True,
          keep_best_k: int | None = None, save_every_steps: int | None = None,
          log_every: int = 50, tb: bool = True):
    """Official recipe: Adam(lr, wd=1e-4), weighted CE, RawBoost on train; keep a checkpoint
    whenever val loss (w/ or w/o aug) improves. Returns history list.

    Extras for preemptible notebooks (Colab/Kaggle):
      resume=True          continue from ``out_dir/last_state.pth`` (model + Adam + scaler + history)
      save_every_steps=N   also refresh last_state.pth every N steps inside an epoch; on resume the
                           interrupted epoch runs only its remaining steps (with a new shuffle)
      keep_best_k=K        delete epoch checkpoints outside the K best (saves disk / Drive quota)

    Logging (see nes2net/logutil.py): every ``log_every`` steps a line with loss, lr, throughput,
    data-loading share, GPU memory and epoch ETA -> console + out_dir/train.log; the same values
    plus per-epoch val loss / val EER go to out_dir/metrics.jsonl and TensorBoard (out_dir/tb).
    ``log`` (a callable) is kept for backward compatibility and receives the text lines too.
    """
    os.makedirs(out_dir, exist_ok=True)
    R = RunLogger(out_dir, tb=tb)
    _ext = log

    def log(msg):
        R.info(msg)
        if _ext is not None:
            _ext(msg)
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
        R.metric('resume', epoch=start_ep, step=start_step)

    n_par = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'start: epochs {start_ep}..{epochs}, {len(tr)} steps/epoch (bs {bs} x accum {accum}), '
        f'train {len(train_items)} / dev {len(dev_items)} utts, lr {lr:g}, RawBoost algo {algo}, '
        f'params {n_par / 1e6:.1f}M ({n_tr / 1e6:.1f}M trainable), device {device}, amp {amp}')
    R.metric('start', lr=lr, bs=bs, accum=accum, algo=algo, steps_per_epoch=len(tr), n_train=len(train_items),
             n_dev=len(dev_items), params_m=n_par / 1e6, trainable_m=n_tr / 1e6)
    t0 = time.time()
    out_of_time = lambda: bool(time_budget_h) and (time.time() - t0) / 3600 > time_budget_h
    for ep in range(start_ep, epochs + 1):
        model.train(); te = time.time()
        run, n = carry if ep == start_ep else (0.0, 0)
        skip = start_step if ep == start_ep else 0
        opt.zero_grad(set_to_none=True)
        stopped = False
        w_loss, w_n, w_data, w_t0 = 0.0, 0, 0.0, time.time()     # logging window
        t_fetch = time.time()
        for step, (x, y, _) in enumerate(tr, start=skip):
            if step >= len(tr):
                break
            w_data += time.time() - t_fetch                       # time spent waiting for the DataLoader
            x, y = x.to(device, non_blocking=True), torch.as_tensor(y).long().to(device)
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp and device.startswith('cuda')):
                out = model(x)
            loss = crit(out.float(), y) / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            li = loss.item() * accum
            run += li * x.size(0); n += x.size(0)
            w_loss += li * x.size(0); w_n += x.size(0)
            if not math.isfinite(li):
                log(f'WARNING non-finite loss {li} at epoch {ep} step {step + 1} (AMP scaler {scaler.get_scale():g})')
            if log_every and (step + 1) % log_every == 0:
                if device.startswith('cuda'):
                    torch.cuda.synchronize()
                wt = time.time() - w_t0
                done, total = step + 1, len(tr)
                rate = (done - skip) / max(time.time() - te, 1e-9)
                eta = (total - done) / max(rate, 1e-9)
                m = dict(epoch=ep, step=done, steps=total, loss=w_loss / max(w_n, 1), loss_epoch=run / max(n, 1),
                         lr=opt.param_groups[0]['lr'], utt_per_s=w_n / max(wt, 1e-9),
                         data_frac=w_data / max(wt, 1e-9), amp_scale=scaler.get_scale() if scaler.is_enabled() else 1.0,
                         eta_epoch_min=eta / 60, elapsed_h=(time.time() - t0) / 3600, **gpu_mem_gb())
                log(f"ep {ep} {done}/{total} | loss {m['loss']:.4f} (ep avg {m['loss_epoch']:.4f}) | "
                    f"lr {m['lr']:.2e} | {m['utt_per_s']:.1f} utt/s | data {100 * m['data_frac']:.0f}% | "
                    f"gpu {m.get('gpu_mem_gb', 0):.1f}/{m.get('gpu_peak_gb', 0):.1f} GB | ETA ep {m['eta_epoch_min']:.1f} min")
                R.metric('train_step', tb_step=(ep - 1) * total + done, **m)
                w_loss, w_n, w_data, w_t0 = 0.0, 0, 0.0, time.time()
            if save_every_steps and (step + 1) % save_every_steps == 0 and step + 1 < len(tr):
                save_state(ep, step + 1, run, n)
                R.metric('checkpoint', epoch=ep, step=step + 1, kind='last_state')
            if out_of_time():
                save_state(ep, step + 1, run, n); stopped = True
                break
            t_fetch = time.time()
        if stopped and n and len(tr) - (step + 1) > 0:
            log(f'time budget reached mid-epoch {ep} (step {step + 1}/{len(tr)}); state saved, rerun to resume')
            R.metric('stop', epoch=ep, step=step + 1, reason='time_budget'); R.close()
            return hist
        t_train = time.time() - te; tv = time.time()
        l0, e0 = val_loss(model, dv0, crit, device, amp, return_eer=True)
        l1, e1 = val_loss(model, dv1, crit, device, amp, return_eer=True) if dv1 is not None else (float('nan'),) * 2
        t_val = time.time() - tv
        save = False
        if l0 <= best0: best0, save = l0, True
        if dv1 is not None and l1 <= best1: best1, save = l1, True
        if save:
            torch.save(model.state_dict(), os.path.join(out_dir, f'epoch_{ep}.pth'))
        h = dict(epoch=ep, train_loss=run / max(n, 1), val_loss_wo_aug=l0, val_loss_w_aug=l1,
                 val_eer_wo_aug=e0, val_eer_w_aug=e1, saved=save, sec=time.time() - te,
                 train_sec=t_train, val_sec=t_val)
        hist.append(h)
        log(f"== epoch {ep}/{epochs} | train loss {h['train_loss']:.4f} | val loss {l0:.4f} / aug {l1:.4f} | "
            f"val EER {e0:.2f}% / aug {e1:.2f}% | best {best0:.4f} / {best1:.4f} | "
            f"{'saved' if save else 'not saved'} | train {t_train / 60:.1f} + val {t_val / 60:.1f} min")
        R.metric('epoch', tb_step=ep, **h, best_val_loss_wo_aug=best0, best_val_loss_w_aug=best1, **gpu_mem_gb())
        with open(os.path.join(out_dir, 'history.json'), 'w') as f:
            json.dump(hist, f, indent=1)
        if keep_best_k:
            _prune(out_dir, hist, keep_best_k)
        save_state(ep, 0, 0.0, 0)
        if out_of_time():
            log(f'time budget reached after epoch {ep}; rerun to resume')
            R.metric('stop', epoch=ep, step=0, reason='time_budget'); break
    else:
        R.metric('finished', epoch=epochs)
    R.close()
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
