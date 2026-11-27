"""Experiment drivers used by the notebook and by ``python -m nes2net.run --preset <name>``.

Each function takes the preset config (``nes2net.presets.get``) and the runtime ``Env``
(``nes2net.env.setup``) and writes its outputs under ``env.work``:

  sanity()          back-end equals official code, params / MACs
  eval_official()   part A   -> partA_results.json, A_*.txt score files
  eval_official_full()  part A (full sets) -> partA_full_results.json, A_full/
  train()           part B   -> runB2/ (checkpoints, last_state.pth, train.log, metrics.jsonl, tb/)
  eval_trained()    part B evaluation -> partB_results.json
  ablation()        part C   -> partC_ablation.csv / .png
  layer_study()     part C2  -> partC_layers.csv
  diag_data()       training-data sanity check -> partDIAG_results.json, DIAG_audio_*.csv
  diag_ckpt()       your trained checkpoint on train/dev subsets in eval/bn_batch/train mode -> partDC_results.json
  diag_train()      training-loop check (front-end loading, BN modes, short runs) -> partDT_results.json
  vi_zeroshot()     Vietnamese zero-shot probe + English control -> partVI_results.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from . import engine
from .env import Env
from .logutil import get_logger

log = get_logger('nes2net.run')
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REF = {
    'ITW (4s first window)': 'Arena same ckpt: 8.48', 'In-the-Wild (4s)': 'Arena same ckpt: 8.48',
    'ITW (full utterance)': 'paper 5.52 best / 6.60 mean (other ckpt)',
    'In-the-Wild (full utt)': 'paper 5.52 best / 6.60 mean (other ckpt)',
    'ASVspoof2019 LA eval': 'Arena same ckpt: 0.13', 'ASVspoof2019 LA eval (4s)': 'Arena same ckpt: 0.13',
    'ASVspoof2021 DF (phase=eval, 10/80 shards)': 'paper 1.65 (this ckpt, all shards)',
    'ASVspoof2021 DF (all trials, 10/80 shards)': 'Arena 3.61 (all 611,829 trials)',
    'ASVspoof2021 LA': 'paper 1.73 (phase=eval) / Arena 6.14 (all trials)',
    'ASVspoof2021 DF': 'paper 1.65 (phase=eval) / Arena 3.61 (all trials)',
}


def _free():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _tag(s: str) -> str:
    return s.replace(':', '_').replace('/', '_').replace('@', '_n').replace(' ', '_').replace('(', '').replace(')', '')


def _official_model(cfg, env):
    from huggingface_hub import hf_hub_download
    from .model import load_official_checkpoint
    ck = hf_hub_download(cfg['ckpt']['repo'], cfg['ckpt']['file'])
    model = load_official_checkpoint(ck).to(env.device)
    log.info('official checkpoint loaded (strict): %s/%s', cfg['ckpt']['repo'], cfg['ckpt']['file'])
    return model


# ------------------------------------------------------------------------------------------ 0
def sanity(env: Env | None = None, run_tests: bool = True) -> pd.DataFrame:
    from .backend import build_backend, count_macs, count_params
    if run_tests and os.path.isdir(os.path.join(REPO_DIR, 'tests')):
        r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                            os.path.join(REPO_DIR, 'tests', 'test_all.py')], capture_output=True, text=True)
        print(r.stdout[-1200:])
    rows = [{'backend': n, 'params_k': round(count_params(build_backend(n)) / 1e3, 1),
             'MMACs_4s': round(count_macs(build_backend(n)), 1)} for n in ('nes2net', 'nes2net_x', 'dr_res2net')]
    print('paper (Table III): Nes2Net 511k / 58.11 MMACs, Nes2Net-X 511k / 91.35 MMACs')
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ A
def eval_official(cfg: dict, env: Env) -> dict:
    from .datasets_zoo import get_items
    model, res, cache = _official_model(cfg, env), {}, {}
    for name, spec, full in cfg['A']['sets']:
        items = cache.setdefault(spec, get_items(spec))
        s = engine.score(model, items, bs=1 if full else 32, device=env.device, full=full,
                         workers=env.workers, desc=name)
        engine.write_scores(s, os.path.join(env.work, f'A_{_tag(name)}.txt'))
        res[name] = engine.evaluate_scores(s, items)
        log.info('%-46s n=%7d EER=%.2f%% minDCF=%.4f', name, res[name]['n'], res[name]['eer'], res[name]['minDCF'])
        json.dump(res, open(os.path.join(env.work, 'partA_results.json'), 'w'), indent=1)
    del model; _free()
    return res


def eval_official_full(cfg: dict, env: Env) -> dict:
    from .datasets_zoo import get_items
    from .full_eval import score_hfbench_full
    model, af = _official_model(cfg, env), os.path.join(env.work, 'A_full')
    os.makedirs(af, exist_ok=True)
    path = os.path.join(env.work, 'partA_full_results.json')
    res = json.load(open(path)) if os.path.exists(path) else {}
    if cfg['A_full']['asv19_test'] and 'ASVspoof2019 LA eval (4s)' not in res:
        items = get_items('asv19:test')
        s = engine.score(model, items, bs=32, device=env.device, workers=env.workers, desc='LA19 eval')
        engine.write_scores(s, os.path.join(af, 'ASVspoof2019_LA_eval_scores.txt'))
        res['ASVspoof2019 LA eval (4s)'] = {'all_trials': engine.evaluate_scores(s, items)}
        json.dump(res, open(path, 'w'), indent=1); del items
    for name, repo, full in cfg['A_full']['hfbench']:
        res[name] = score_hfbench_full(model, repo, os.path.join(af, _tag(name)), device=env.device, bs=32,
                                       workers=env.workers, full_utt=full, shard_dir=os.path.join(env.tmp, 'shards'),
                                       log=log.info)   # resumable: already-scored shards are skipped
        json.dump(res, open(path, 'w'), indent=1)
        log.info('%s: %s', name, {k: round(v['eer'], 3) for k, v in res[name].items() if isinstance(v, dict) and 'eer' in v})
    del model; _free()
    return res


def results_table(res: dict) -> pd.DataFrame:
    """Flatten part-A results (subset or full) into one table with reference numbers."""
    rows = []
    for k, v in res.items():
        subs = {s: v[s] for s in ('phase_eval', 'all_trials') if s in v} if 'eer' not in v else {'': v}
        for sub, r in subs.items():
            rows.append({'set': k, 'subset': sub, 'n': r['n'], 'EER %': round(r['eer'], 2),
                         'minDCF': round(r['minDCF'], 4), 'reference': _REF.get(k, '')})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ B
def _restore_from_kaggle_input(out: str):
    """Kaggle: a kernel can mount its own previous output; copy <out>/ from there to resume."""
    prev = sorted(glob.glob(f'/kaggle/input/**/{os.path.basename(out)}/last_state.pth', recursive=True))
    if prev and not os.path.exists(os.path.join(out, 'last_state.pth')):
        src = os.path.dirname(prev[0]); os.makedirs(out, exist_ok=True)
        for f in os.listdir(src):
            p = os.path.join(src, f)
            (shutil.copytree if os.path.isdir(p) else shutil.copy2)(p, os.path.join(out, f))
        log.info('restored previous session state from %s', src)


def resolve_grad_ckpt(setting, device: str = 'cuda') -> bool:
    """'auto': gradient checkpointing only on GPUs with < 22 GB (T4 16 GB needs it for bs 12 at 4 s;
    A100 / L4 / A10 do not). It trades ~30 % extra compute for memory and does not change the math."""
    if setting != 'auto':
        return bool(setting)
    if not (device.startswith('cuda') and torch.cuda.is_available()):
        return False
    return torch.cuda.get_device_properties(0).total_memory / 2**30 < 22


def _build_train_model(cfg, device='cuda'):
    from .model import SSLNes2Net
    b = cfg['B']
    gc = resolve_grad_ckpt(b.get('grad_ckpt', 'auto'), device)
    return SSLNes2Net(b['ssl'], backend=b['backend'], gradient_checkpointing=gc,
                      attn_implementation=b.get('attn', 'sdpa'), ssl_dropout=b.get('ssl_dropout', 'fairseq'))


def train(cfg: dict, env: Env):
    from .datasets_zoo import get_items
    b, out = cfg['B'], os.path.join(env.work, cfg['B']['out'])
    if env.platform == 'kaggle' and b['resume_from_kaggle_input']:
        _restore_from_kaggle_input(out)
    engine.set_seed(b['seed'])
    model = _build_train_model(cfg, env.device)
    tr, dv = get_items(b['train']), get_items(b['dev'])
    hist = engine.train(model, tr, dv, out, epochs=b['epochs'], lr=b['lr'], bs=b['bs'], algo=b['algo'],
                        device=env.device, amp=True, workers=env.workers, time_budget_h=b['time_budget_h'],
                        resume=True, keep_best_k=b['keep_best_k'], save_every_steps=b['save_every_steps'],
                        log_every=b['log_every'], eval_bs=b.get('eval_bs'))
    log.info('%d/%d epochs done; state in %s (rerun to resume)', len(hist), b['epochs'], out)
    del model; _free()
    return hist


def eval_trained(cfg: dict, env: Env) -> dict:
    """Average the best checkpoints of part B (by val loss w/o aug) and score the eval sets."""
    from .datasets_zoo import get_items
    from .model import SSLNes2Net
    b, out = cfg['B'], os.path.join(env.work, cfg['B']['out'])
    hist = json.load(open(os.path.join(out, 'history.json')))
    best = engine.best_epochs(hist, k=b['avg_k'])
    assert best, 'no finished epoch yet: run training first'
    model = SSLNes2Net(pretrained_ssl=False, backend=b['backend'])
    engine.average_checkpoints(model, [os.path.join(out, f'epoch_{e}.pth') for e in best],
                               os.path.join(out, 'avg.pth'))
    model.to(env.device)
    res = {'averaged_epochs': best}
    for name, spec in b['eval_sets']:
        items = get_items(spec)
        s = engine.score(model, items, device=env.device, workers=env.workers, desc=name)
        res[name] = engine.evaluate_scores(s, items)
        log.info('B %s EER=%.2f%%', name, res[name]['eer'])
    json.dump(res, open(os.path.join(env.work, 'partB_results.json'), 'w'), indent=1)
    del model; _free()
    return res


def plot_train_logs(run_dir: str, show_tb: bool = False, tail: int = 3000):
    """Loss / throughput / val EER from metrics.jsonl + the end of train.log."""
    import matplotlib.pyplot as plt
    from .logutil import load_metrics
    mp = os.path.join(run_dir, 'metrics.jsonl')
    if not os.path.exists(mp):
        print('no metrics yet at', mp); return None
    M = load_metrics(mp)
    st, ep = M[M.event == 'train_step'], M[M.event == 'epoch']
    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    if len(st):
        x = (st.epoch - 1) + st.step / st.steps
        ax[0].plot(x, st.loss, lw=.7, label='train loss (window)'); ax[0].set_xlabel('epoch')
        ax[1].plot(x, st.utt_per_s, lw=.7, label='utt/s'); ax[1].plot(x, 100 * st.data_frac, lw=.7, label='% waiting for data')
    if len(ep):
        ax[0].plot(ep.epoch, ep.val_loss_wo_aug, 'o-', label='val loss')
        ax[0].plot(ep.epoch, ep.val_loss_w_aug, 's-', label='val loss (aug)')
        ax[2].plot(ep.epoch, ep.val_eer_wo_aug, 'o-', label='val EER %')
        ax[2].plot(ep.epoch, ep.val_eer_w_aug, 's-', label='val EER % (aug)')
    for a in ax:
        a.legend(); a.grid(alpha=.3)
    plt.tight_layout(); plt.show()
    lp = os.path.join(run_dir, 'train.log')
    if os.path.exists(lp):
        print(open(lp).read()[-tail:])
    if show_tb:
        from IPython import get_ipython
        get_ipython().run_line_magic('load_ext', 'tensorboard')
        get_ipython().run_line_magic('tensorboard', f'--logdir {os.path.join(run_dir, "tb")}')
    return M


# ------------------------------------------------------------------------------------------ C
def _features(spec_list, layers, cache_root, ssl_name, env):
    from transformers import AutoModel
    from . import ablation as A
    from .datasets_zoo import get_items
    ssl = AutoModel.from_pretrained(ssl_name)
    dirs = {s: A.extract_features(ssl, get_items(s), os.path.join(cache_root, _tag(s)), layers=layers,
                                  device=env.device, workers=env.workers) for s in spec_list}
    del ssl; _free()
    return dirs


def ablation(cfg: dict, env: Env) -> pd.DataFrame:
    from . import ablation as A
    c = cfg['C']
    dirs = _features([c['train'], c['dev']] + c['evals'], c['layers'], os.path.join(env.tmp, 'feat'), c['ssl'], env)
    return A.run_grid(c['groups'], dirs[c['train']], dirs[c['dev']], {_tag(s): dirs[s] for s in c['evals']},
                      seeds=c['seeds'], out_csv=os.path.join(env.work, 'partC_ablation.csv'),
                      layers=c['layers'], epochs=c['epochs'], device=env.device)


def plot_ablation(df: pd.DataFrame, env: Env):
    import matplotlib.pyplot as plt
    evc = [c for c in df.columns if c.endswith('_eer') and c != 'dev_eer']
    fig, axes = plt.subplots(1, len(evc), figsize=(7 * len(evc), 6))
    axes = np.atleast_1d(axes)
    lab = df['group'].str[:2] + ' ' + df['cfg'].str.replace('"backend": ', '').str.replace('"', '')
    for ax, c in zip(axes, evc):
        g = df.assign(lab=lab).groupby('lab')[c]
        m, s = g.mean(), g.std()
        ax.barh(m.index, m.values, xerr=s.values, color='#4C72B0')
        ax.set_title(f'{c} (mean ± std over seeds)'); ax.set_xlabel('EER %'); ax.invert_yaxis()
    plt.tight_layout(); plt.savefig(os.path.join(env.work, 'partC_ablation.png'), dpi=120); plt.show()


def layer_study(cfg: dict, env: Env) -> pd.DataFrame:
    from . import ablation as A
    L = cfg['L']
    specs = {'train': L['train'], 'dev': L['dev'], 'itw': L['itw']}
    d = _features(list(specs.values()), L['layers'], os.path.join(env.tmp, 'featL'), cfg['C']['ssl'], env)
    d = {k: d[s] for k, s in specs.items()}
    rows = []
    for layers in [[l] for l in L['layers']] + [L['layers']]:
        for sd in L['seeds']:
            r = A.train_backend({'backend': 'nes2net_x'}, d['train'], d['dev'], {'itw': d['itw']},
                                layers=layers, seed=sd, epochs=L['epochs'], device=env.device)
            r['layers'] = f'mix{L["layers"]}' if len(layers) > 1 else f'L{layers[0]}'
            rows.append(r); log.info('layer %s seed %d: ITW EER %.2f%%', r['layers'], sd, r['itw_eer'])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(env.work, 'partC_layers.csv'), index=False)
    return df


# ------------------------------------------------------------------------------------------ VI
def vi_zeroshot(cfg: dict, env: Env) -> dict:
    """Official checkpoint, zero-shot, on text-matched FLEURS-vs-TTS probes (see nes2net/tts_probe.py)."""
    from .tts_probe import build_probe
    v = cfg['VI']
    path = os.path.join(env.work, 'partVI_results.json')
    res = {}
    probes = {}
    for name, lang, tts in v['probes']:                 # synthesise first, then load the detector
        probes[name] = build_probe(lang, tts, os.path.join(env.tmp, 'tts_probe', lang), split=v['split'],
                                   limit=v['limit'], device=env.device, log=log.info)
        _free()
    model = _official_model(cfg, env)
    for name, sets in probes.items():
        for variant, items in sets.items():
            for full in ([False, True] if v['full_utt'] else [False]):
                tag = f'{name} | {variant} | {"full utt" if full else "4s"}'
                s = engine.score(model, items, bs=1 if full else 32, device=env.device, full=full,
                                 workers=env.workers, desc=tag)
                engine.write_scores(s, os.path.join(env.work, f'VI_{_tag(name)}_{variant}_{"full" if full else "4s"}.txt'))
                r = engine.evaluate_scores(s, items, by_attack=False)
                for g in ('male', 'female'):
                    sub = [it for it in items if it.label == 0 or str(it.meta.get('gender', '')).lower() == g]
                    sub_b = [it for it in sub if it.label == 1]
                    if sub_b:
                        r[f'eer_bona_{g}'] = engine.evaluate_scores(s, sub, by_attack=False)['eer']
                sb = np.array([s[it.key] for it in items if it.label == 1])
                ss = np.array([s[it.key] for it in items if it.label == 0])
                r['score_mean_bona'], r['score_mean_spoof'] = float(sb.mean()), float(ss.mean())
                res[tag] = r
                log.info('%s: EER %.2f%% (n=%d)', tag, r['eer'], r['n'])
                json.dump(res, open(path, 'w'), indent=1)
    del model; _free()
    return res


def vi_table(res: dict) -> pd.DataFrame:
    rows = []
    for k, r in res.items():
        probe, variant, proto = [x.strip() for x in k.split('|')]
        rows.append({'probe': probe, 'audio': variant, 'window': proto, 'n': r['n'], 'EER %': round(r['eer'], 2),
                     'minDCF': round(r['minDCF'], 4), 'mean score bona': round(r['score_mean_bona'], 2),
                     'mean score TTS': round(r['score_mean_spoof'], 2)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ DIAG
def _audio_stats(src) -> dict:
    """Stored format + simple signal statistics of one utterance (as the model sees it, 16 kHz)."""
    import io
    import soundfile as sf
    from .data import HFRef, load_audio
    raw = src.read() if isinstance(src, HFRef) else src
    if isinstance(raw, (bytes, bytearray)):
        info = sf.info(io.BytesIO(raw))
    else:
        info = sf.info(raw)
    x = load_audio(raw)
    a = np.abs(x)
    thr = 10 ** (-40 / 20) * (a.max() + 1e-9)                # -40 dB relative to the peak
    nz = np.flatnonzero(a > thr)
    lead = (nz[0] if len(nz) else len(x)) / 16000
    trail = ((len(x) - 1 - nz[-1]) if len(nz) else len(x)) / 16000
    return {'format': f'{info.format}/{info.subtype}', 'sr_stored': info.samplerate, 'dur_s': len(x) / 16000,
            'rms_db': float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)), 'peak': float(a.max()),
            'lead_sil_s': float(lead), 'trail_sil_s': float(trail), 'dc': float(x.mean())}


def diag_data(cfg: dict, env: Env) -> dict:
    """Is the training data what we think it is? Compares train / dev / test of the ASVspoof2019 LA
    copy: (1) label vs attack-id crosstab (bona fide must be attack '-'), (2) per split x class audio
    statistics (format, sample rate, duration, level, leading/trailing silence), (3) EER of the
    official checkpoint on a random subset of each split. If the official model fails on train or dev
    while it passes on test, the copy (not the training code) is the problem."""
    from .data import balanced_subset
    from .datasets_zoo import get_items
    d = cfg['DIAG']
    out = {'crosstab': {}, 'stats': {}, 'official_eer': {}}
    subsets = {}
    for name, spec in d['specs']:
        items = get_items(spec)
        ct = pd.crosstab(pd.Series([it.label for it in items], name='label (1=bona)'),
                         pd.Series([str((it.meta or {}).get('attack', '-')) for it in items], name='attack'))
        out['crosstab'][name] = {str(k): {str(c): int(v) for c, v in row.items() if v} for k, row in ct.iterrows()}
        log.info('%s: %d utts, bona %d / spoof %d', name, len(items), sum(it.label == 1 for it in items),
                 sum(it.label == 0 for it in items))
        sub = balanced_subset(items, d['n'], seed=d['seed'], bona_frac=0.5)
        subsets[name] = sub
        rows = []
        for it in sub:
            st = _audio_stats(it.source); st['label'] = it.label; rows.append(st)
        df = pd.DataFrame(rows)
        agg = df.groupby('label').agg(n=('dur_s', 'size'), dur_mean=('dur_s', 'mean'), dur_p50=('dur_s', 'median'),
                                      rms_db=('rms_db', 'mean'), peak=('peak', 'mean'),
                                      lead_sil=('lead_sil_s', 'mean'), trail_sil=('trail_sil_s', 'mean'),
                                      dc=('dc', 'mean'))
        fmts = df.groupby('label')['format'].agg(lambda s: dict(s.value_counts()))
        srs = df.groupby('label')['sr_stored'].agg(lambda s: dict(s.value_counts()))
        out['stats'][name] = {str(k): {**{c: float(v) for c, v in r.items()}, 'formats': fmts[k], 'sr_stored': {str(a): int(b) for a, b in srs[k].items()}}
                              for k, r in agg.iterrows()}
        df.to_csv(os.path.join(env.work, f'DIAG_audio_{name}.csv'), index=False)
    model = _official_model(cfg, env)
    for name, sub in subsets.items():
        s = engine.score(model, sub, bs=32, device=env.device, workers=env.workers, desc=f'official on {name}')
        out['official_eer'][name] = engine.evaluate_scores(s, sub)
        log.info('official checkpoint on %s subset: EER %.2f%%', name, out['official_eer'][name]['eer'])
    del model; _free()
    json.dump(out, open(os.path.join(env.work, 'partDIAG_results.json'), 'w'), indent=1, default=str)
    return out


def diag_tables(out: dict):
    """(EER per split, audio stats per split x class, label/attack crosstab) as DataFrames."""
    eer = pd.DataFrame([{'split': k, 'n': v['n'], 'official EER %': round(v['eer'], 2)} for k, v in out['official_eer'].items()])
    st = pd.DataFrame([{'split': sp, 'label': 'bona' if lab == '1' else 'spoof',
                        **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}}
                       for sp, d in out['stats'].items() for lab, r in d.items()])
    ct = pd.DataFrame([{'split': sp, 'label': lab, **cols} for sp, d in out['crosstab'].items() for lab, cols in d.items()]).fillna(0)
    return eer, st, ct


# ------------------------------------------------------------------------------------------ DIAG_TRAIN
_BN = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)


def _set_mode(model, mode: str):
    """'eval': normal inference. 'bn_batch': eval, but BatchNorm normalises with the batch's own
    statistics (momentum 0 -> running stats untouched). 'train': full train mode (dropout on too)."""
    if mode == 'train':
        model.train()
    else:
        model.eval()
    for m in model.modules():
        if isinstance(m, _BN):
            if mode == 'eval':
                m.momentum = getattr(m, '_mom0', m.momentum)
            else:
                m._mom0 = getattr(m, '_mom0', m.momentum)
                m.train(); m.momentum = 0.0


@torch.no_grad()
def _score_mode(model, items, mode, env, bs=24, amp=True):
    _set_mode(model, mode)
    out = {}
    for x, y, k in engine._loader(items, bs, workers=env.workers):
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp and env.device.startswith('cuda')):
            s = model(x.to(env.device, non_blocking=True))[:, 1].float().cpu().numpy()
        out.update(zip(k, s.tolist()))
    _set_mode(model, 'eval')
    return out


def _mode_eers(model, subsets: dict, env, modes=('eval', 'bn_batch', 'train')) -> dict:
    res = {}
    for name, items in subsets.items():
        for mode in modes:
            r = engine.evaluate_scores(_score_mode(model, items, mode, env), items, by_attack=False)
            res[f'{name}/{mode}'] = round(r['eer'], 2)
            log.info('  EER %-6s %-9s %6.2f%%', name, mode, r['eer'])
    return res


def _pos_conv_check(model, ssl_name: str) -> dict:
    """Compare the positional-conv weight the front-end actually uses with the weight stored in the
    hub file (weight_g * weight_v / ||weight_v||, weight-norm over all dims but 2). A large difference
    means the pretrained weights were not loaded and the layer is random."""
    from huggingface_hub import hf_hub_download, list_repo_files
    files = list_repo_files(ssl_name)
    fn = 'model.safetensors' if 'model.safetensors' in files else 'pytorch_model.bin'
    p = hf_hub_download(ssl_name, fn)
    if fn.endswith('.safetensors'):
        from safetensors.torch import load_file
        sd = load_file(p)
    else:
        sd = torch.load(p, map_location='cpu', weights_only=False)
    g = next((v for k, v in sd.items() if k.endswith('pos_conv_embed.conv.weight_g')
              or k.endswith('pos_conv_embed.conv.parametrizations.weight.original0')), None)
    v = next((v for k, v in sd.items() if k.endswith('pos_conv_embed.conv.weight_v')
              or k.endswith('pos_conv_embed.conv.parametrizations.weight.original1')), None)
    if g is None or v is None:
        return {'file': fn, 'error': 'pos_conv weight_g/weight_v not found in file'}
    v = v.float()
    w_file = v * (g.float() / v.norm(dim=(0, 1), keepdim=True))
    w_model = model.encoder.pos_conv_embed.conv.weight.detach().float().cpu()
    d = (w_model - w_file).abs().max().item()
    return {'file': fn, 'max_abs_diff': d, 'w_file_absmean': w_file.abs().mean().item(),
            'w_model_absmean': w_model.abs().mean().item(), 'loaded_ok': d < 1e-4 * max(w_file.abs().max().item(), 1)}


def diag_train(cfg: dict, env: Env) -> dict:
    """Is the training loop (engine.train) broken? Small, controlled runs on ASVspoof2019 LA:

      0. front-end check: XLS-R weights reported missing by transformers + direct comparison of the
         positional conv with the hub file.
      1. official checkpoint: EER on balanced train / dev subsets in three modes (eval, BatchNorm with
         batch statistics, full train mode). All should be ~0 %.
      2. variants (name, init, epochs, overrides) trained with the real engine.train (paper recipe on a
         random train subset); afterwards the same three-mode EERs. Reading:
           * 'official' init degrades after 1 epoch     -> the loop damages a good model (code bug)
           * train-subset EER good, dev bad             -> genuine over-fitting / data difference
           * eval-mode EER bad but bn_batch good        -> BatchNorm running statistics are the problem
           * AMP variant bad, fp32 variant fine         -> fp16 numerics
    """
    from .data import balanced_subset
    from .datasets_zoo import get_items
    from .model import load_official_checkpoint
    from huggingface_hub import hf_hub_download
    d, b = cfg['DT'], cfg['B']
    root = os.path.join(env.work, 'diag_train'); os.makedirs(root, exist_ok=True)
    out = {'config': d, 'variants': {}}
    dump = lambda: json.dump(out, open(os.path.join(env.work, 'partDT_results.json'), 'w'), indent=1, default=str)
    rng = np.random.default_rng(d['seed'])
    shuf = lambda xs: [xs[i] for i in rng.permutation(len(xs))]     # mixed classes in every BN batch
    tr_all, dv_all = get_items(d['train']), get_items(d['dev'])
    tr = balanced_subset(tr_all, d['n_train'], seed=d['seed'])                     # natural class ratio
    dv = balanced_subset(dv_all, d['n_dev'], seed=d['seed'])
    tr_keys = {it.key for it in tr}
    score_sets = {'train': shuf(balanced_subset([it for it in tr_all if it.key in tr_keys], d['n_score'],
                                                seed=d['seed'] + 1, bona_frac=0.5)),
                  'dev': shuf(balanced_subset(dv_all, d['n_score'], seed=d['seed'] + 1, bona_frac=0.5))}
    log.info('DT: train subset %d (bona %d), dev subset %d, score sets %s', len(tr), sum(i.label for i in tr),
             len(dv), {k: len(v) for k, v in score_sets.items()})

    # 0. front-end loading
    m = _build_train_model(cfg, env.device)
    li = m.ssl_model.loading_info or {}
    out['frontend'] = {'missing_keys': list(li.get('missing_keys', [])), 'unexpected_keys': list(li.get('unexpected_keys', []))[:20],
                       'mismatched_keys': [str(x) for x in li.get('mismatched_keys', [])]}
    try:
        out['frontend']['pos_conv'] = _pos_conv_check(m.ssl_model.model, b['ssl'])
    except Exception as e:                                       # never let the check kill the run
        out['frontend']['pos_conv'] = {'error': repr(e)}
    log.info('DT front-end: %s', out['frontend'])
    del m; _free(); dump()

    # 1. official checkpoint
    ck = hf_hub_download(cfg['ckpt']['repo'], cfg['ckpt']['file'])
    gc_on = resolve_grad_ckpt(b.get('grad_ckpt', 'auto'), env.device)
    m = load_official_checkpoint(ck, gradient_checkpointing=gc_on).to(env.device)
    log.info('DT official checkpoint:')
    out['official'] = _mode_eers(m, score_sets, env)
    del m; _free(); dump()

    # 2. variants
    for name, init, epochs, ov in d['variants']:
        log.info('DT variant %s: init=%s epochs=%d overrides=%s', name, init, epochs, ov)
        engine.set_seed(d['seed'])
        m = load_official_checkpoint(ck, gradient_checkpointing=gc_on) if init == 'official' else _build_train_model(cfg, env.device)
        kw = dict(epochs=epochs, lr=d['lr'], bs=d['bs'], algo=d['algo'], device=env.device, amp=True,
                  workers=env.workers, resume=False, keep_best_k=1, log_every=50, tb=False, eval_bs=b.get('eval_bs'))
        kw.update(ov)
        vdir = os.path.join(root, name)
        shutil.rmtree(vdir, ignore_errors=True)
        hist = engine.train(m, tr, dv, vdir, **kw)
        r = {'init': init, 'overrides': ov, 'history': hist}
        log.info('DT %s after training (final weights):', name)
        r['eer'] = _mode_eers(m, score_sets, env)
        out['variants'][name] = r
        for f in glob.glob(os.path.join(vdir, '*.pth')):        # 1.2 GB each; logs are kept
            os.remove(f)
        del m; _free(); dump()
    return out


def diag_ckpt(cfg: dict, env: Env) -> dict:
    """EER of one of YOUR trained checkpoints on balanced train / dev subsets, in eval, bn_batch and
    train mode. Train good + dev bad in eval mode = genuine over-fitting; train bad in eval mode but
    good in train mode = a train/eval mismatch (dropout / BatchNorm)."""
    from .data import balanced_subset
    from .datasets_zoo import get_items
    from .model import SSLNes2Net
    d = cfg['DC']
    path = d['path'] if os.path.isabs(d['path']) else os.path.join(env.work, d['path'])
    sd = torch.load(path, map_location='cpu', weights_only=False)
    sd = sd.get('model', sd) if isinstance(sd, dict) else sd          # last_state.pth or a plain state_dict
    m = SSLNes2Net(pretrained_ssl=False, backend=cfg['B']['backend'])
    m.load_state_dict(sd)
    m.to(env.device)
    rng = np.random.default_rng(d['seed'])
    shuf = lambda xs: [xs[i] for i in rng.permutation(len(xs))]
    subs = {name: shuf(balanced_subset(get_items(spec), d['n'], seed=d['seed'], bona_frac=0.5))
            for name, spec in d['specs']}
    log.info('DC %s:', path)
    out = {'path': path, 'eer': _mode_eers(m, subs, env)}
    json.dump(out, open(os.path.join(env.work, 'partDC_results.json'), 'w'), indent=1)
    del m; _free()
    return out


def diag_train_table(out: dict) -> pd.DataFrame:
    rows = [{'model': 'official ckpt', **out.get('official', {})}]
    for name, r in out.get('variants', {}).items():
        last = r['history'][-1] if r['history'] else {}
        rows.append({'model': name, **r['eer'], 'train_loss': round(last.get('train_loss', float('nan')), 4),
                     'val_loss': round(last.get('val_loss_wo_aug', float('nan')), 4)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ CLI
def run_all(cfg: dict, env: Env, display=print):
    """Run every enabled part in order (what the notebook does cell by cell)."""
    R = cfg['run']
    if R['sanity']:
        display(sanity(env))
    if R['A']:
        display(results_table(eval_official(cfg, env)))
    if R['A_full']:
        display(results_table(eval_official_full(cfg, env)))
    if R['B']:
        train(cfg, env)
    if R['B_eval']:
        display(eval_trained(cfg, env))
    if R['C']:
        from .ablation import summarize
        display(summarize(ablation(cfg, env)))
    if R.get('DIAG'):
        for t in diag_tables(diag_data(cfg, env)):
            display(t)
    if R.get('DT'):
        display(diag_train_table(diag_train(cfg, env)))
    if R.get('DC'):
        display(pd.Series(diag_ckpt(cfg, env)['eer'], name='EER %'))
    if R.get('VI'):
        display(vi_table(vi_zeroshot(cfg, env)))
    if R['L']:
        df = layer_study(cfg, env)
        display(df.groupby('layers')[['dev_eer', 'itw_eer']].agg(['mean', 'std']).round(2))


def main(argv=None):
    from . import presets
    from .env import setup
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--preset', default='default', choices=presets.names())
    p.add_argument('--set', action='append', default=[], metavar='PART.KEY=JSON',
                   help='override, e.g. --set B.lr=2e-6 --set run.C=false')
    p.add_argument('--no-drive', action='store_true')
    a = p.parse_args(argv)
    ov = {}
    for kv in a.set:
        k, v = kv.split('=', 1)
        part, key = k.split('.', 1)
        ov.setdefault(part, {})[key] = json.loads(v)
    env = setup(use_drive=not a.no_drive)
    print(env)
    cfg = presets.get(a.preset, env.platform, ov)
    presets.show(cfg)
    run_all(cfg, env)


if __name__ == '__main__':
    main()
