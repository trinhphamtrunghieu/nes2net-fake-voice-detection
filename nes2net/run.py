"""Experiment drivers used by the notebook and by ``python -m nes2net.run --preset <name>``.

Each function takes the preset config (``nes2net.presets.get``) and the runtime ``Env``
(``nes2net.env.setup``) and writes its outputs under ``env.work``:

  sanity()          back-end equals official code, params / MACs
  eval_official()   part A   -> partA_results.json, A_*.txt score files
  eval_official_full()  part A (full sets) -> partA_full_results.json, A_full/
  train()           part B   -> runB/ (checkpoints, last_state.pth, train.log, metrics.jsonl, tb/)
  eval_trained()    part B evaluation -> partB_results.json
  ablation()        part C   -> partC_ablation.csv / .png
  layer_study()     part C2  -> partC_layers.csv
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
    """Kaggle: a kernel can mount its own previous output; copy runB/ from there to resume."""
    prev = sorted(glob.glob('/kaggle/input/**/runB/last_state.pth', recursive=True))
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
                      attn_implementation=b.get('attn', 'sdpa'))


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
