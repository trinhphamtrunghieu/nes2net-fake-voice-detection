"""Nes2Net / Nes2Net-X back-ends (Liu et al., IEEE T-IFS 2025, arXiv:2504.05657).

Faithful re-implementation of the official code
(github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW, model_scripts/wav2vec2_Nes2Net_X.py and
github.com/Liu-Tianchi/Nes2Net, models/WavLM_Nes2Net*.py).

Parameter / module names are kept identical to the official code so the released
checkpoints load with ``strict=True``.  Extra keyword arguments are *ablation switches*;
with their defaults the network is bit-for-bit the official one.

Shapes: input (B, C=1024, T) frame-level SSL features -> logits (B, n_out).
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


class SEModule(nn.Module):
    """Squeeze-and-excitation over channels (time-averaged)."""

    def __init__(self, channels: int, SE_ratio: int = 8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // SE_ratio, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(channels // SE_ratio, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class Bottle2neck(nn.Module):
    """Inner Res2Net block (one per outer group).

    weighted_sum=False -> Nes2Net   (classic Res2Net: sp = sp + x_i, Conv1d)
    weighted_sum=True  -> Nes2Net-X (concat previous output with x_i along a new axis,
                                     Conv2d with (k,1) kernel, then a learnable weighted sum)
    Ablation switches:
      use_se=False        drop the SE module
      inner_hier=False    no hierarchical (cumulative) connection inside the block,
                          i.e. every split is processed independently (plain group conv)
    """

    def __init__(self, inplanes, planes, kernel_size=3, dilation=2, scale=8, SE_ratio=8,
                 weighted_sum=False, use_se=True, inner_hier=True):
        super().__init__()
        width = int(math.floor(planes / scale))
        self.weighted = weighted_sum
        self.inner_hier = inner_hier
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        num_pad = math.floor(kernel_size / 2) * dilation
        convs, bns, ws = [], [], []
        for i in range(self.nums):
            if weighted_sum:
                convs.append(nn.Conv2d(width, width, kernel_size=(kernel_size, 1),
                                       dilation=(dilation, 1), padding=(num_pad, 0)))
                bns.append(nn.BatchNorm2d(width))
                ws.append(nn.Parameter(torch.ones(1, 1, 1, i + 2) * (1 / (i + 2))))
            else:
                convs.append(nn.Conv1d(width, width, kernel_size=kernel_size,
                                       dilation=dilation, padding=num_pad))
                bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        if weighted_sum:
            self.weighted_sum = nn.ParameterList(ws)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes, SE_ratio) if use_se else nn.Identity()

    def _forward_plain(self, out):
        spx = torch.split(out, self.width, 1)
        outs = []
        sp = None
        for i in range(self.nums):
            sp = spx[i] if (i == 0 or not self.inner_hier) else sp + spx[i]
            sp = self.bns[i](self.relu(self.convs[i](sp)))
            outs.append(sp)
        outs.append(spx[self.nums])
        return torch.cat(outs, 1)

    def _forward_weighted(self, out):
        out = out.unsqueeze(-1)  # B C T 1
        spx = torch.split(out, self.width, 1)
        sp = spx[self.nums]
        outs = []
        for i in range(self.nums):
            if self.inner_hier:
                sp = torch.cat((sp, spx[i]), -1)            # B w T (i+2)
            else:
                sp = torch.cat([spx[i]] * (i + 2), -1)
            sp = self.bns[i](self.relu(self.convs[i](sp)))
            outs.append(torch.sum(sp * self.weighted_sum[i], dim=-1))
        outs.append(spx[self.nums].squeeze(-1))
        return torch.cat(outs, 1)

    def forward(self, x):
        residual = x
        out = self.bn1(self.relu(self.conv1(x)))
        out = self._forward_weighted(out) if self.weighted else self._forward_plain(out)
        out = self.bn3(self.relu(self.conv3(out)))
        out = self.se(out)
        return out + residual


class ASTP(nn.Module):
    """Attentive statistics pooling (ECAPA-TDNN)."""

    def __init__(self, in_dim, bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)

    def forward(self, x):
        alpha = torch.softmax(self.linear2(torch.tanh(self.linear1(x))), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
        return torch.cat([mean, torch.sqrt(var.clamp(min=1e-10))], dim=1)


class Nested_Res2Net_TDNN(nn.Module):
    """Outer Res2Net over channel groups; each group is processed by an inner Bottle2neck.

    Official configuration: Nes_ratio=[8, 8], SE_ratio=[1], dilation=2, pool_func='mean'.
    Ablation switches:
      outer_hier=False   remove the outer cumulative connection (sp = sp + x_i)
      keep_last=False    also process the last outer group instead of passing it through
    """

    def __init__(self, Nes_ratio: Sequence[int] = (8, 8), input_channel: int = 1024,
                 n_output_logits: int = 2, dilation: int = 2, pool_func: str = 'mean',
                 SE_ratio: Sequence[int] = (1,), weighted_sum: bool = True,
                 use_se: bool = True, inner_hier: bool = True, outer_hier: bool = True,
                 kernel_size: int = 3):
        super().__init__()
        self.Nes_ratio = Nes_ratio[0]
        assert input_channel % Nes_ratio[0] == 0
        C = input_channel // Nes_ratio[0]
        self.C = C
        self.outer_hier = outer_hier
        self.Build_in_Res2Nets = nn.ModuleList([
            Bottle2neck(C, C, kernel_size=kernel_size, dilation=dilation, scale=Nes_ratio[1],
                        SE_ratio=SE_ratio[0], weighted_sum=weighted_sum, use_se=use_se,
                        inner_hier=inner_hier)
            for _ in range(Nes_ratio[0] - 1)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(C) for _ in range(Nes_ratio[0] - 1)])
        self.bn = nn.BatchNorm1d(input_channel)
        self.relu = nn.ReLU()
        self.pool_func = pool_func
        if pool_func == 'mean':
            self.fc = nn.Linear(input_channel, n_output_logits)
        elif pool_func == 'ASTP':
            self.pooling = ASTP(in_dim=input_channel, bottleneck_dim=128)
            self.fc = nn.Linear(2 * input_channel, n_output_logits)
        else:
            raise ValueError(pool_func)

    def embed(self, x):
        spx = torch.split(x, self.C, 1)
        outs, sp = [], None
        for i in range(self.Nes_ratio - 1):
            sp = spx[i] if (i == 0 or not self.outer_hier) else sp + spx[i]
            sp = self.bns[i](self.relu(self.Build_in_Res2Nets[i](sp)))
            outs.append(sp)
        outs.append(spx[-1])
        out = self.relu(self.bn(torch.cat(outs, 1)))
        return out.mean(-1) if self.pool_func == 'mean' else self.pooling(out)

    def forward(self, x):
        return self.fc(self.embed(x))


# ----------------------------------------------------------------------------------
# Baselines used in the paper's comparison (simplified, for back-end ablations)
# ----------------------------------------------------------------------------------
class DR_Res2Net(nn.Module):
    """Conventional pipeline: dimensionality-reduction layer (1024 -> D) + one Res2Net block.
    Mirrors the paper's 'Res2Net with DR layer' baseline (Table III/IV)."""

    def __init__(self, input_channel=1024, D=128, scale=4, n_output_logits=2, dilation=2):
        super().__init__()
        self.dr = nn.Sequential(nn.Conv1d(input_channel, D, 1), nn.ReLU(), nn.BatchNorm1d(D))
        self.block = Bottle2neck(D, D, kernel_size=3, dilation=dilation, scale=scale,
                                 SE_ratio=8, weighted_sum=False)
        self.fc = nn.Linear(D, n_output_logits)

    def forward(self, x):
        return self.fc(self.block(self.dr(x)).mean(-1))


class MeanLinear(nn.Module):
    """Weakest sensible baseline: temporal mean + linear."""

    def __init__(self, input_channel=1024, n_output_logits=2):
        super().__init__()
        self.fc = nn.Linear(input_channel, n_output_logits)

    def forward(self, x):
        return self.fc(x.mean(-1))


def build_backend(name: str = 'nes2net_x', input_channel: int = 1024, **kw) -> nn.Module:
    name = name.lower()
    if name in ('nes2net', 'nes2net_x'):
        kw.setdefault('weighted_sum', name == 'nes2net_x')
        return Nested_Res2Net_TDNN(input_channel=input_channel, **kw)
    if name == 'dr_res2net':
        return DR_Res2Net(input_channel=input_channel, **kw)
    if name == 'mean_linear':
        return MeanLinear(input_channel=input_channel, **kw)
    raise ValueError(name)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def count_macs(m: nn.Module, C: int = 1024, T: int = 201) -> float:
    """Approximate MACs (in millions) for conv/linear layers via forward hooks.
    T=201 frames corresponds to a 4 s (64,600-sample) input at 20 ms hop."""
    macs = [0]

    def hook(mod, inp, out):
        if isinstance(mod, (nn.Conv1d, nn.Conv2d)):
            k = mod.weight[0].numel()  # in/groups * kernel
            macs[0] += out.numel() // out.shape[0] * k
        elif isinstance(mod, nn.Linear):
            macs[0] += mod.in_features * mod.out_features

    hs = [mm.register_forward_hook(hook) for mm in m.modules()
          if isinstance(mm, (nn.Conv1d, nn.Conv2d, nn.Linear))]
    was_training = m.training
    m.eval()
    with torch.no_grad():
        m(torch.randn(1, C, T))
    m.train(was_training)
    for h in hs:
        h.remove()
    return macs[0] / 1e6
