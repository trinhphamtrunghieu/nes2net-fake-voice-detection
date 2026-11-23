"""Nes2Net reproduction (arXiv:2504.05657)."""
from .backend import Nested_Res2Net_TDNN, Bottle2neck, build_backend, count_params, count_macs
from .model import SSLNes2Net, load_official_checkpoint, convert_official_state_dict

__version__ = "0.1.0"
