"""Score a model on a full SpeechAntiSpoofingBenchmarks set shard-by-shard (bounded disk/RAM),
then report EER on (a) the official ASVspoof2021 phase == 'eval' subset and (b) all trials."""
from __future__ import annotations

import json
import os

from . import engine
from .data import Item
from .datasets_zoo import hfbench_files, load_hfbench_shard


def score_hfbench_full(model, repo: str, out_dir: str, device='cuda', bs=32, workers=4,
                       full_utt=False, shard_dir: str | None = None, log=print):
    os.makedirs(out_dir, exist_ok=True)
    tag = repo.split('/')[-1]
    score_path = os.path.join(out_dir, f'{tag}_scores.txt')
    meta_path = os.path.join(out_dir, f'{tag}_meta.jsonl')
    done = set()
    if os.path.exists(os.path.join(out_dir, f'{tag}_done.txt')):
        done = set(open(os.path.join(out_dir, f'{tag}_done.txt')).read().split())
    files = hfbench_files(repo)
    for i, f in enumerate(files):
        if f in done:
            continue
        items, p = load_hfbench_shard(repo, f, shard_dir or os.path.join(out_dir, '_shards'))
        s = engine.score(model, items, bs=1 if full_utt else bs, device=device, workers=workers,
                         full=full_utt, desc=f'{tag} {i + 1}/{len(files)}')
        with open(score_path, 'a') as fs, open(meta_path, 'a') as fm:
            for it in items:
                fs.write(f'{it.key} {s[it.key]}\n')
                fm.write(json.dumps({'key': it.key, 'label': it.label,
                                     **{k: it.meta.get(k) for k in ('attack', 'subset', 'codec', 'vocoder')}}) + '\n')
        os.remove(p)
        with open(os.path.join(out_dir, f'{tag}_done.txt'), 'a') as fd:
            fd.write(f + '\n')
        log(f'{tag}: shard {i + 1}/{len(files)} scored ({len(items)} trials)')
    return summarize(score_path, meta_path)


def summarize(score_path, meta_path):
    scores = dict(l.split() for l in open(score_path))
    scores = {k: float(v) for k, v in scores.items()}
    metas = [json.loads(l) for l in open(meta_path)]
    items = [Item(m['key'], None, m['label'], m) for m in metas]
    res = {'all_trials': engine.evaluate_scores(scores, items)}
    ev = [it for it in items if it.meta.get('subset') == 'eval']
    if ev:
        res['phase_eval'] = engine.evaluate_scores(scores, ev)
    by = {}
    bona = [it for it in ev if it.label == 1]
    for key in ('codec', 'vocoder'):
        for v in sorted({it.meta.get(key) for it in ev if it.meta.get(key) and it.label == 0}):
            # codec is a channel condition shared by bona fide and spoof trials (official
            # per-condition EER uses both); vocoder only describes spoofs -> pool all bona fide
            b_ = [it for it in bona if it.meta.get(key) == v] if key == 'codec' else bona
            sp = [it for it in ev if it.label == 0 and it.meta.get(key) == v]
            if b_ and sp:
                by[f'{key}={v}'] = engine.evaluate_scores(scores, b_ + sp, by_attack=False)['eer']
    if by:
        res['phase_eval_breakdown_eer'] = by
    return res
