"""Run presets: one dict per experiment mode, selected with ``PRESET = '...'`` in the notebook
or ``python -m nes2net.run --preset ...``.

  smoke        tiny subsets of every part, ~30 min on a T4: checks that the pipeline runs
  default      official-checkpoint eval on subsets + budget end-to-end training + ablations
  full_eval    official checkpoint on the FULL test sets + full ablation grid (~6-8 h on a T4)
  vi_zeroshot  official checkpoint, zero-shot on a Vietnamese probe (FLEURS vi vs MMS-TTS vi) + an
               English control probe built the same way (~30-45 min on a T4)
  diag_data    sanity check of the ASVspoof2019 LA copy used for training: labels vs attack ids,
               audio statistics per split/class, official-checkpoint EER on train/dev/test (~15 min)
  paper_train  the paper's training recipe (lr 2.5e-7, bs 12, 100 epochs, RawBoost 4, full dev),
               run over many sessions with resume

Only ``time_budget_h`` depends on the platform (session limits); everything else is shared.
Override anything with a nested dict, e.g. ``get('default', 'colab', {'B': {'lr': 2e-6}})``.
"""
from __future__ import annotations

import copy
import json

ABLATION_GROUPS = ['A0_baselines', 'A1_nest_ratio', 'A2_se_ratio', 'A3_pooling',
                   'A4_dilation_kernel', 'A5_connections']

_BASE = {
    # which parts run
    'run': {'sanity': True, 'A': False, 'A_full': False, 'B': False, 'B_eval': False, 'C': False, 'L': False,
            'VI': False, 'DIAG': False},
    'ckpt': {'repo': 'SpeechAntiSpoofingBenchmarks/Nes2Net', 'file': 'nes2net_x_DF1.65.pth'},
    # A: official checkpoint on subsets. sets: [name, spec, full_utterance]
    'A': {'sets': [
        ['ITW (4s first window)', 'hfbench:InTheWild', False],
        ['ITW (full utterance)', 'hfbench:InTheWild', True],
        ['ASVspoof2019 LA eval', 'asv19:test', False],
        ['ASVspoof2021 DF (phase=eval, 10/80 shards)', 'hfbench:ASVspoof2021_DF:10:eval', False],
        ['ASVspoof2021 DF (all trials, 10/80 shards)', 'hfbench:ASVspoof2021_DF:10', False],
    ]},
    # A_full: official checkpoint on complete sets (hfbench sets are scored shard by shard)
    'A_full': {'asv19_test': True, 'hfbench': [
        ['In-the-Wild (4s)', 'InTheWild', False],
        ['In-the-Wild (full utt)', 'InTheWild', True],
        ['ASVspoof2021 LA', 'ASVspoof2021_LA', False],
        ['ASVspoof2021 DF', 'ASVspoof2021_DF', False],
    ]},
    # B: end-to-end training (official recipe except lr / dev size / time budget, see docs)
    'B': {'ssl': 'facebook/wav2vec2-xls-r-300m', 'backend': 'nes2net_x', 'seed': 12345,
          'train': 'asv19:train', 'dev': 'asv19:validation@6000',
          'lr': 1e-6, 'bs': 12, 'epochs': 100, 'algo': 4, 'time_budget_h': None,
          'keep_best_k': 3, 'save_every_steps': 500, 'avg_k': 3, 'log_every': 50,
          'out': 'runB', 'resume_from_kaggle_input': True,
          # speed settings (do not change the recipe): gradient checkpointing only when VRAM < 22 GB,
          # fused SDPA attention, bigger batch for forward-only validation
          'grad_ckpt': 'auto', 'attn': 'sdpa', 'eval_bs': 48,
          'eval_sets': [['ASVspoof2019 LA eval', 'asv19:test'], ['ITW', 'hfbench:InTheWild'],
                        ['DF eval-phase 10 shards', 'hfbench:ASVspoof2021_DF:10:eval']]},
    # C: frozen-feature back-end ablations
    'C': {'ssl': 'facebook/wav2vec2-xls-r-300m', 'train': 'asv19:train@6000', 'dev': 'asv19:validation@2000',
          'evals': ['asv19:test@4000', 'hfbench:InTheWild@4000'], 'layers': [24], 'seeds': [0, 1, 2],
          'epochs': 15, 'groups': ABLATION_GROUPS},
    # VI: zero-shot cross-lingual probe (FLEURS read speech vs the same sentences from MMS-TTS).
    # probes: [name, FLEURS lang, TTS model]; English control built the same way.
    'VI': {'probes': [['vi (FLEURS vs MMS-TTS)', 'vi_vn', 'facebook/mms-tts-vie'],
                      ['en control (FLEURS vs MMS-TTS)', 'en_us', 'facebook/mms-tts-eng']],
           'split': 'test', 'limit': None, 'full_utt': True},
    # DIAG: data sanity for the training data (label/attack crosstab, audio statistics per split and
    # class, official-checkpoint EER per split). n = random utterances per split for audio stats + EER.
    'DIAG': {'specs': [['train', 'asv19:train'], ['dev', 'asv19:validation'], ['test', 'asv19:test']],
             'n': 2000, 'seed': 0},
    # L: SSL layer study (cache several layers on smaller subsets)
    'L': {'layers': [8, 12, 18, 24], 'train': 'asv19:train@3000', 'dev': 'asv19:validation@1000',
          'itw': 'hfbench:InTheWild@2000', 'seeds': [0, 1, 2], 'epochs': 15},
}

_PRESETS = {
    'default': {'run': {'A': True, 'C': True, 'L': True}},
    'smoke': {
        'run': {'A': True, 'B': True, 'B_eval': True, 'C': True, 'L': True, 'VI': True, 'DIAG': True},
        'A': {'sets': [['ITW@600 (4s)', 'hfbench:InTheWild@600', False],
                       ['ASVspoof2019 LA eval@600', 'asv19:test@600', False],
                       ['DF phase=eval (1 shard)@600', 'hfbench:ASVspoof2021_DF:1:eval@600', False]]},
        'B': {'train': 'asv19:train@400', 'dev': 'asv19:validation@200', 'bs': 8, 'epochs': 2,
              'time_budget_h': 0.25, 'keep_best_k': 1, 'save_every_steps': 20, 'avg_k': 1, 'log_every': 10,
              'eval_sets': [['LA19 eval@300', 'asv19:test@300'], ['ITW@300', 'hfbench:InTheWild@300']]},
        'C': {'train': 'asv19:train@600', 'dev': 'asv19:validation@300',
              'evals': ['asv19:test@300', 'hfbench:InTheWild@300'], 'seeds': [0], 'epochs': 3,
              'groups': ['A0_baselines', 'A5_connections']},
        'L': {'train': 'asv19:train@300', 'dev': 'asv19:validation@150', 'itw': 'hfbench:InTheWild@200',
              'seeds': [0], 'epochs': 3},
        'VI': {'limit': 40},
        'DIAG': {'n': 100},
    },
    'full_eval': {'run': {'A_full': True, 'C': True, 'L': True}},
    'vi_zeroshot': {'run': {'VI': True}},
    'diag_data': {'run': {'DIAG': True}},
    'paper_train': {
        'run': {'B': True},
        'B': {'lr': 2.5e-7, 'bs': 12, 'epochs': 100, 'algo': 4, 'dev': 'asv19:validation',
              'keep_best_k': 3, 'avg_k': 3},
    },
}

# per-session time limits (h): Kaggle 12 h, Colab free often disconnects, Colab Pro ~24 h with tab open
_BUDGET = {
    'default': {'kaggle': 8.3, 'colab': 3.5, 'local': None},
    'paper_train': {'kaggle': 11.0, 'colab': 11.5, 'local': None},
}


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def names() -> list[str]:
    return list(_PRESETS)


def get(preset: str = 'default', platform: str = 'local', overrides: dict | None = None) -> dict:
    if preset not in _PRESETS:
        raise ValueError(f'unknown preset {preset!r}; choose from {names()}')
    cfg = _merge(_BASE, _PRESETS[preset])
    if cfg['B']['time_budget_h'] is None and preset in _BUDGET:
        cfg['B']['time_budget_h'] = _BUDGET[preset].get(platform)
    cfg = _merge(cfg, overrides or {})
    cfg['preset'], cfg['platform'] = preset, platform
    return cfg


def show(cfg: dict) -> str:
    on = [k for k, v in cfg['run'].items() if v]
    s = f"preset={cfg['preset']} platform={cfg['platform']} parts={on}\n"
    for part in ('B',) if cfg['run']['B'] else ():
        b = cfg['B']
        s += (f"  B: lr={b['lr']:g} bs={b['bs']} epochs={b['epochs']} algo={b['algo']} dev={b['dev']} "
              f"time_budget_h={b['time_budget_h']} keep_best_k={b['keep_best_k']}\n")
    print(s.rstrip())
    return s


def dumps(cfg: dict) -> str:
    return json.dumps(cfg, indent=1)
