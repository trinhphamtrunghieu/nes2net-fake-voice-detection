"""Logging helpers: human-readable log (console + file), JSONL metrics, optional TensorBoard.

Files written by ``RunLogger(out_dir)``:
  out_dir/train.log       timestamped text log (same lines as the console)
  out_dir/metrics.jsonl   one JSON object per event: {"t", "event", ...}  (easy to load with pandas)
  out_dir/tb/             TensorBoard event files (only if tensorboard is installed and tb=True)
All writes are appends, so a resumed run continues the same files.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time

_FMT = '%(asctime)s %(levelname)-5s %(name)s | %(message)s'


def get_logger(name: str = 'nes2net', log_file: str | None = None, level=logging.INFO) -> logging.Logger:
    """Idempotent: calling twice does not duplicate handlers (safe in notebooks)."""
    lg = logging.getLogger(name)
    lg.setLevel(level)
    lg.propagate = False
    if not any(getattr(h, '_nes2net_console', False) for h in lg.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(_FMT, '%H:%M:%S'))
        h._nes2net_console = True
        lg.addHandler(h)
    if log_file:
        log_file = os.path.abspath(log_file)
        if not any(getattr(h, 'baseFilename', None) == log_file for h in lg.handlers):
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter(_FMT, '%Y-%m-%d %H:%M:%S'))
            lg.addHandler(fh)
    return lg


class RunLogger:
    def __init__(self, out_dir: str, name: str = 'nes2net.train', tb: bool = True):
        os.makedirs(out_dir, exist_ok=True)
        self.log = get_logger(name, os.path.join(out_dir, 'train.log'))
        self.metrics_path = os.path.join(out_dir, 'metrics.jsonl')
        self.tb = None
        if tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(os.path.join(out_dir, 'tb'))
            except Exception:          # tensorboard not installed -> JSONL only
                self.tb = None

    def info(self, msg, *a):
        self.log.info(msg, *a)

    def metric(self, event: str, tb_step: int | None = None, **values):
        clean = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in values.items()}
        rec = {'t': round(time.time(), 3), 'event': event, **clean}     # strict JSON: NaN/inf -> null (jq-friendly)
        with open(self.metrics_path, 'a') as f:
            f.write(json.dumps(rec, default=float, allow_nan=False) + '\n')
        if self.tb is not None and tb_step is not None:
            for k, v in values.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                    self.tb.add_scalar(f"{event}/{k}", v, tb_step)

    def close(self):
        if self.tb is not None:
            self.tb.flush(); self.tb.close()


def gpu_mem_gb() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {'gpu_mem_gb': round(torch.cuda.memory_allocated() / 2**30, 2),
                    'gpu_peak_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    except Exception:
        pass
    return {}


def load_metrics(path: str):
    """metrics.jsonl -> pandas DataFrame."""
    import pandas as pd
    return pd.DataFrame([json.loads(l) for l in open(path)])
