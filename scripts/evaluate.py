#!/usr/bin/env python
"""Score a checkpoint on one or more test sets and report EER / minDCF / per-attack EER.

  python scripts/evaluate.py --ckpt runs/x/avg_top5.pth --sets asv19:test hfbench:InTheWild
  python scripts/evaluate.py --official nes2net_x_DF1.65.pth --sets hfbench:ASVspoof2021_DF::eval
Protocol: --full scores whole utterances (paper's ITW protocol); default is the 4 s first window.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from nes2net.model import SSLNes2Net, load_official_checkpoint
from nes2net.datasets_zoo import get_items
from nes2net import engine

p = argparse.ArgumentParser()
g = p.add_mutually_exclusive_group(required=True)
g.add_argument('--ckpt', help='checkpoint trained with this repo'); g.add_argument('--official', help='official fairseq-based .pth')
p.add_argument('--sets', nargs='+', required=True); p.add_argument('--out', default='scores')
p.add_argument('--backend', default='nes2net_x'); p.add_argument('--full', action='store_true')
p.add_argument('--batch_size', type=int, default=32); p.add_argument('--no_amp', action='store_true')
a = p.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
if a.official:
    model = load_official_checkpoint(a.official, backend=a.backend)
else:
    model = SSLNes2Net(pretrained_ssl=False, backend=a.backend)
    model.load_state_dict(torch.load(a.ckpt, map_location='cpu'))
model.to(dev)
os.makedirs(a.out, exist_ok=True); allres = {}
for spec in a.sets:
    items = get_items(spec)
    s = engine.score(model, items, bs=a.batch_size, device=dev, full=a.full, amp=not a.no_amp, desc=spec)
    tag = spec.replace(':', '_').replace('/', '_')
    engine.write_scores(s, os.path.join(a.out, f'{tag}.txt'))
    allres[spec] = engine.evaluate_scores(s, items)
    print(spec, json.dumps({k: v for k, v in allres[spec].items() if k != 'per_attack_eer'}))
json.dump(allres, open(os.path.join(a.out, 'results.json'), 'w'), indent=1)
