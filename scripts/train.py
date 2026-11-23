#!/usr/bin/env python
"""End-to-end training: XLS-R (fine-tuned) + Nes2Net(-X), ASVspoof2019 LA, RawBoost.

Paper / official recipe (Nes2Net_ASVspoof_ITW/README):
  --lr 2.5e-7 --batch_size 12 --epochs 100 --algo 4 --weight_decay 1e-4, weighted CE [0.1, 0.9]
Kaggle-budget recipe (one T4/P100 session):
  --lr 1e-6 --batch_size 12 --epochs 100 --time_budget_h 8.5 --amp
Example:
  python scripts/train.py --train asv19:train --dev asv19:validation --out runs/x --time_budget_h 8.5
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from nes2net.model import SSLNes2Net
from nes2net.datasets_zoo import get_items
from nes2net import engine

p = argparse.ArgumentParser()
p.add_argument('--train', default='asv19:train'); p.add_argument('--dev', default='asv19:validation')
p.add_argument('--out', default='runs/nes2net_x')
p.add_argument('--ssl', default='facebook/wav2vec2-xls-r-300m'); p.add_argument('--ssl_layer', default='last')
p.add_argument('--backend', default='nes2net_x', choices=['nes2net', 'nes2net_x', 'dr_res2net', 'mean_linear'])
p.add_argument('--Nes_ratio', type=int, nargs=2, default=[8, 8]); p.add_argument('--SE_ratio', type=int, default=1)
p.add_argument('--dilation', type=int, default=2); p.add_argument('--pool_func', default='mean')
p.add_argument('--freeze_ssl', action='store_true')
p.add_argument('--lr', type=float, default=1e-6); p.add_argument('--weight_decay', type=float, default=1e-4)
p.add_argument('--batch_size', type=int, default=12); p.add_argument('--accum', type=int, default=1)
p.add_argument('--epochs', type=int, default=100); p.add_argument('--algo', type=int, default=4)
p.add_argument('--no_val_aug', action='store_true'); p.add_argument('--seed', type=int, default=12345)
p.add_argument('--amp', action='store_true'); p.add_argument('--grad_ckpt', action='store_true')
p.add_argument('--workers', type=int, default=4); p.add_argument('--time_budget_h', type=float, default=None)
p.add_argument('--avg_k', type=int, default=5, help='average the k best checkpoints at the end')
a = p.parse_args()

engine.set_seed(a.seed)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
kw = {}
if a.backend.startswith('nes2net'):
    kw = dict(Nes_ratio=a.Nes_ratio, SE_ratio=[a.SE_ratio], dilation=a.dilation, pool_func=a.pool_func)
model = SSLNes2Net(a.ssl, ssl_layer=a.ssl_layer, backend=a.backend, freeze_ssl=a.freeze_ssl,
                   gradient_checkpointing=a.grad_ckpt, **kw)
tr, dv = get_items(a.train), get_items(a.dev)
print(f'train {len(tr)}  dev {len(dv)}  params total {sum(p.numel() for p in model.parameters())/1e6:.1f}M')
os.makedirs(a.out, exist_ok=True); json.dump(vars(a), open(os.path.join(a.out, 'args.json'), 'w'), indent=1)
hist = engine.train(model, tr, dv, a.out, epochs=a.epochs, lr=a.lr, wd=a.weight_decay, bs=a.batch_size,
                    algo=a.algo, val_aug=not a.no_val_aug, device=dev, amp=a.amp, accum=a.accum,
                    workers=a.workers, time_budget_h=a.time_budget_h)
best = engine.best_epochs(hist, k=a.avg_k)
print('best epochs (val loss w/o aug):', best)
engine.average_checkpoints(model, [os.path.join(a.out, f'epoch_{e}.pth') for e in best],
                           os.path.join(a.out, f'avg_top{len(best)}.pth'))
torch.save(torch.load(os.path.join(a.out, f'epoch_{best[0]}.pth')), os.path.join(a.out, 'best.pth'))
