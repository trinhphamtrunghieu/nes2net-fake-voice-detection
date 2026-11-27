"""Full SSL + Nes2Net model and loader for the official checkpoints."""
from __future__ import annotations

import re

import torch
import torch.nn as nn

from .backend import build_backend
from .frontend import HFSSL


class SSLNes2Net(nn.Module):
    """wav2vec2 XLS-R (or WavLM) front-end + Nes2Net(-X) back-end.

    Submodule names mirror the official code: ``ssl_model.model`` and ``Nested_Res2Net_TDNN``.
    Output: 2 logits, index 1 = bona fide (official convention). Score = logits[:, 1].
    """

    def __init__(self, ssl_name='facebook/wav2vec2-xls-r-300m', ssl_layer='last',
                 backend='nes2net_x', pretrained_ssl=True, freeze_ssl=False,
                 gradient_checkpointing=False, attn_implementation='sdpa', ssl_dropout='fairseq', **backend_kw):
        super().__init__()
        self.ssl_model = HFSSL(ssl_name, layer=ssl_layer, pretrained=pretrained_ssl,
                               gradient_checkpointing=gradient_checkpointing,
                               attn_implementation=attn_implementation, dropout=ssl_dropout)
        self.freeze_ssl = freeze_ssl
        if freeze_ssl:
            for p in self.ssl_model.model.parameters():
                p.requires_grad_(False)
        backend_kw.setdefault('n_output_logits', 2)
        self.Nested_Res2Net_TDNN = build_backend(backend, input_channel=self.ssl_model.out_dim,
                                                 **backend_kw)

    def forward(self, wav):
        if self.freeze_ssl:
            with torch.no_grad():
                self.ssl_model.model.eval()
                feat = self.ssl_model(wav)
        else:
            feat = self.ssl_model(wav)
        return self.Nested_Res2Net_TDNN(feat)


# ----------------------------------------------------------------------------------
# fairseq -> HF key conversion for the released wav2vec2 + Nes2Net-X checkpoints
# ----------------------------------------------------------------------------------
_LAYER_MAP = [
    (r'self_attn\.k_proj', 'attention.k_proj'),
    (r'self_attn\.v_proj', 'attention.v_proj'),
    (r'self_attn\.q_proj', 'attention.q_proj'),
    (r'self_attn\.out_proj', 'attention.out_proj'),
    (r'self_attn_layer_norm', 'layer_norm'),
    (r'fc1', 'feed_forward.intermediate_dense'),
    (r'fc2', 'feed_forward.output_dense'),
    (r'final_layer_norm', 'final_layer_norm'),
]
_DROP = ('quantizer.', 'project_q.', 'final_proj.', 'label_embs_concat', 'target_glu')


def fairseq_key_to_hf(k: str, pos_conv_names: dict[str, str]) -> str | None:
    """Map one fairseq Wav2Vec2Model key (without prefix) to the HF Wav2Vec2Model key."""
    if k.startswith(_DROP):
        return None
    if k == 'mask_emb':
        return 'masked_spec_embed'
    if k.startswith('post_extract_proj.'):
        return k.replace('post_extract_proj.', 'feature_projection.projection.')
    if k.startswith('layer_norm.'):
        return k.replace('layer_norm.', 'feature_projection.layer_norm.')
    if k.startswith('encoder.layer_norm.'):
        return k
    m = re.match(r'feature_extractor\.conv_layers\.(\d+)\.(\d+)(?:\.(\d+))?\.(weight|bias)$', k)
    if m:
        i, a, b, p = m.groups()
        if a == '0':
            return f'feature_extractor.conv_layers.{i}.conv.{p}'
        if a == '2' and b == '1':      # layer_norm mode: Sequential(TransposeLast, LN, TransposeLast)
            return f'feature_extractor.conv_layers.{i}.layer_norm.{p}'
        if a == '2' and b is None:     # group_norm mode (base models)
            return f'feature_extractor.conv_layers.{i}.layer_norm.{p}'
    m = re.match(r'encoder\.pos_conv\.0\.(weight_g|weight_v|bias|parametrizations\.weight\.original[01])$', k)
    if m:
        g = {'parametrizations.weight.original0': 'weight_g',
             'parametrizations.weight.original1': 'weight_v'}.get(m.group(1), m.group(1))
        return pos_conv_names[g]
    m = re.match(r'encoder\.layers\.(\d+)\.(.+)$', k)
    if m:
        n, rest = m.groups()
        for pat, rep in _LAYER_MAP:
            if re.match(pat + r'\.', rest):
                return f'encoder.layers.{n}.' + re.sub('^' + pat, rep, rest)
    raise KeyError(f'Unmapped fairseq key: {k}')


def _pos_conv_names(hf_keys) -> dict[str, str]:
    base = 'encoder.pos_conv_embed.conv.'
    if base + 'weight_g' in hf_keys:
        return {'weight_g': base + 'weight_g', 'weight_v': base + 'weight_v', 'bias': base + 'bias'}
    return {'weight_g': base + 'parametrizations.weight.original0',
            'weight_v': base + 'parametrizations.weight.original1', 'bias': base + 'bias'}


def convert_official_state_dict(sd: dict, model: SSLNes2Net) -> dict:
    """Convert an official wav2vec2_Nes2Net_X state_dict (fairseq front-end) to our model."""
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    hf_keys = set(model.ssl_model.model.state_dict().keys())
    pc = _pos_conv_names(hf_keys)
    out = {}
    for k, v in sd.items():
        if k.startswith('ssl_model.model.'):
            nk = fairseq_key_to_hf(k[len('ssl_model.model.'):], pc)
            # mask_emb is only used for pre-training masking; HF omits masked_spec_embed when
            # mask_time_prob == mask_feature_prob == 0 (our setting), so drop it in that case.
            if nk is not None and (nk != 'masked_spec_embed' or nk in hf_keys):
                out['ssl_model.model.' + nk] = v
        else:
            out[k] = v
    return out


def load_official_checkpoint(path: str, map_location='cpu', **model_kw) -> SSLNes2Net:
    """Build the paper's wav2vec2 + Nes2Net-X and load a released .pth (strict)."""
    model_kw.setdefault('backend', 'nes2net_x')
    model = SSLNes2Net(pretrained_ssl=False, **model_kw)
    sd = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd and not any(k.startswith('ssl_model') for k in sd):
        sd = sd['state_dict']
    conv = convert_official_state_dict(sd, model)
    missing, unexpected = model.load_state_dict(conv, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'Checkpoint mismatch. missing={missing[:10]} unexpected={unexpected[:10]}')
    return model.eval()
