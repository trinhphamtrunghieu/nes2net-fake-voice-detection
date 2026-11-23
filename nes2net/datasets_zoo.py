"""Named dataset getters used by the scripts / notebook.

Spec strings:
  asv19:<split>                 Bisher/ASVspoof_2019_LA on the HF Hub (train|validation|test)
  asv19dir:<root>:<split>       official LA folder (root/ASVspoof2019_LA_{split}/flac + cm protocols)
  itw:<root>                    In-the-Wild release folder (meta.csv + wav)
  hfbench:<repo>[:<n_shards>[:<phase>]]  SpeechAntiSpoofingBenchmarks/* parquet, e.g.
        hfbench:InTheWild, hfbench:ASVspoof2021_DF:10:eval (first 10 of 80 shards, official phase)
  asv21dir:<flac_dir>:<trial_metadata.txt>[:<phase>]   official 2021 eval + keys (phase=eval)
Optional suffix "@N" keeps a random subset of N items (seed 0), e.g. asv19:train@6000.
"""
from __future__ import annotations

import os

from .data import (Item, balanced_subset, hf_items, load_asvspoof_protocol, load_hf_parquet,
                   load_in_the_wild)


def _hfbench(repo: str, n_shards: int | None = None, phase: str | None = None, cache_dir=None):
    """SpeechAntiSpoofingBenchmarks parquet repos: data/test-*.parquet with columns
    path / audio{bytes} / label (ClassLabel 0=bonafide, 1=spoof) / notes (JSON string).
    For ASVspoof2021 LA/DF, notes['subset'] is the official phase (progress|eval|hidden_track);
    the official scoring scripts report EER on phase == 'eval' only."""
    import json
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type='dataset')
                   if f.startswith('data/test-') and f.endswith('.parquet'))
    if n_shards:
        files = files[:n_shards]
    paths = [hf_hub_download(repo, f, repo_type='dataset', cache_dir=cache_dir) for f in files]
    ds = load_dataset('parquet', data_files=paths, split='train', cache_dir=cache_dir)

    def meta(r):
        try:
            m = json.loads(r.get('notes') or '{}')
        except Exception:
            m = {}
        m['attack'] = m.get('attack_id', '-'); m['path'] = r.get('path')
        return m

    def lab(r):
        v = r['label']
        return int(v == 0) if isinstance(v, int) else int(str(v).lower().startswith('bona'))

    def key(r, j):
        try:
            u = json.loads(r.get('notes') or '{}').get('utterance_id')
        except Exception:
            u = None
        return u or os.path.splitext(os.path.basename(r.get('path') or f'row{j}'))[0]

    keep = (lambda m: m.get('subset') in (None, phase)) if phase else None
    return hf_items(ds, lab, key, meta, keep=keep)


def get_items(spec: str, cache_dir=None) -> list[Item]:
    n = None
    if '@' in spec:
        spec, n = spec.rsplit('@', 1); n = int(n)
    kind, _, rest = spec.partition(':')
    if kind == 'asv19':
        items = load_hf_parquet('Bisher/ASVspoof_2019_LA', rest or 'train', cache_dir=cache_dir)
        for it in items:
            it.meta['attack'] = it.meta.get('system_id', '-')
    elif kind == 'asv19dir':
        root, split = rest.rsplit(':', 1)
        tag = {'train': 'train.trn', 'dev': 'dev.trl', 'eval': 'eval.trl'}[split]
        items = load_asvspoof_protocol(
            os.path.join(root, 'ASVspoof2019_LA_cm_protocols', f'ASVspoof2019.LA.cm.{tag}.txt'),
            os.path.join(root, f'ASVspoof2019_LA_{split}', 'flac'))
    elif kind == 'itw':
        items = load_in_the_wild(rest)
    elif kind == 'hfbench':
        parts = rest.split(':')
        repo = parts[0] if '/' in parts[0] else 'SpeechAntiSpoofingBenchmarks/' + parts[0]
        ns = int(parts[1]) if len(parts) > 1 and parts[1] else None
        ph = parts[2] if len(parts) > 2 else None
        items = _hfbench(repo, ns, ph, cache_dir)
    elif kind == 'asv21dir':
        parts = rest.split(':')
        items = load_asvspoof_protocol(parts[1], parts[0], phase=parts[2] if len(parts) > 2 else 'eval')
    else:
        raise ValueError(spec)
    if n:
        items = balanced_subset(items, n, seed=0)
    return items
