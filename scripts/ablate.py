#!/usr/bin/env python
"""Frozen-SSL back-end ablations (see nes2net/ablation.py and docs/ablation_and_gaps.md).

  python scripts/ablate.py --cache /tmp/feat --train asv19:train@6000 --dev asv19:validation@2000 \
      --eval asv19:test@4000 hfbench:InTheWild@4000 --groups A0_baselines A1_nest_ratio --seeds 0 1 2
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from transformers import AutoModel
from nes2net.datasets_zoo import get_items
from nes2net import ablation

p = argparse.ArgumentParser()
p.add_argument('--cache', default='/tmp/nes2net_feat'); p.add_argument('--ssl', default='facebook/wav2vec2-xls-r-300m')
p.add_argument('--layers', type=int, nargs='+', default=[24])
p.add_argument('--train', default='asv19:train@6000'); p.add_argument('--dev', default='asv19:validation@2000')
p.add_argument('--eval', nargs='+', default=['asv19:test@4000', 'hfbench:InTheWild@4000'])
p.add_argument('--groups', nargs='+', default=list(ablation.ABLATIONS))
p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2]); p.add_argument('--epochs', type=int, default=15)
p.add_argument('--out_csv', default='ablation_results.csv')
a = p.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
ssl = AutoModel.from_pretrained(a.ssl)
tag = lambda s: s.replace(':', '_').replace('/', '_').replace('@', '_n')
dirs = {}
for s in [a.train, a.dev] + a.eval:
    dirs[s] = ablation.extract_features(ssl, get_items(s), os.path.join(a.cache, tag(s)), layers=a.layers, device=dev)
del ssl; torch.cuda.empty_cache()
df = ablation.run_grid(a.groups, dirs[a.train], dirs[a.dev], {tag(s): dirs[s] for s in a.eval},
                       seeds=a.seeds, out_csv=a.out_csv, layers=a.layers, epochs=a.epochs, device=dev)
print(ablation.summarize(df).to_string())
