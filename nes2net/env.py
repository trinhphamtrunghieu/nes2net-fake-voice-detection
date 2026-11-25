"""Runtime environment detection: Kaggle / Colab / local.

The only things that differ between platforms are where files go, how many CPU workers to
use and the per-session time limit; everything else is identical, so one notebook serves all.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass


def detect_platform() -> str:
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') or os.path.isdir('/kaggle/working'):
        return 'kaggle'
    if 'google.colab' in sys.modules or os.environ.get('COLAB_RELEASE_TAG'):
        return 'colab'
    return 'local'


@dataclass
class Env:
    platform: str
    work: str        # results / checkpoints (persistent: /kaggle/working, Google Drive, ./work)
    tmp: str         # large scratch: datasets, HF cache, feature caches (lost after the session)
    workers: int     # DataLoader workers
    device: str

    def __str__(self):
        return (f'platform={self.platform} device={self.device} workers={self.workers}\n'
                f'  WORK={self.work}\n  TMP ={self.tmp}')


def setup(use_drive: bool = True, work: str | None = None, tmp: str | None = None,
          max_workers: int = 12) -> Env:
    """Create the directories, point the HF cache at TMP, mount Google Drive on Colab."""
    import torch
    plat = detect_platform()
    if plat == 'kaggle':
        w, t = '/kaggle/working', '/tmp/nes2net'          # /kaggle/working is kept (<= ~20 GB)
    elif plat == 'colab':
        if use_drive:
            from google.colab import drive                # noqa: F401  (only exists on Colab)
            if not os.path.isdir('/content/drive/MyDrive'):
                drive.mount('/content/drive')
            w = '/content/drive/MyDrive/nes2net'
        else:
            w = '/content/nes2net_work'
        t = '/content/nes2net_tmp'                        # local disk: fast, big caches stay off Drive
    else:
        w, t = os.path.abspath('work'), os.path.abspath('tmp')
    w, t = work or w, tmp or t
    os.makedirs(w, exist_ok=True); os.makedirs(t, exist_ok=True)
    os.environ.setdefault('HF_HOME', os.path.join(t, 'hf'))   # never cache GBs of data on Drive
    nw = max(1, min(max_workers, os.cpu_count() or 2))
    return Env(plat, w, t, nw, 'cuda' if torch.cuda.is_available() else 'cpu')
