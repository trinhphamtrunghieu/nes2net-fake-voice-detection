"""SSL front-ends via HuggingFace ``transformers`` (no fairseq needed).

The paper's speech models use wav2vec 2.0 XLS-R 300M (fairseq ``xlsr2_300m.pt``), fine-tuned
end-to-end; the frame features are the *last* transformer layer output (after the final
LayerNorm), shape (B, T, 1024) with T ~= 50 frames/s.  ``facebook/wav2vec2-xls-r-300m`` on
the HF Hub is the same checkpoint converted to HF format.

Differences vs. fairseq handled here:
  * HF applies SpecAugment-style time masking in train mode when ``mask_time_prob>0``;
    fairseq was called with ``mask=False`` -> we set ``mask_time_prob=0``.
  * no input normalisation (the official code feeds the raw waveform).
"""
from __future__ import annotations

import torch
import torch.nn as nn

XLSR_300M_CONFIG = dict(
    hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096,
    hidden_act='gelu', feat_extract_norm='layer', feat_extract_activation='gelu',
    conv_dim=(512,) * 7, conv_stride=(5, 2, 2, 2, 2, 2, 2), conv_kernel=(10, 3, 3, 3, 3, 2, 2),
    conv_bias=True, num_conv_pos_embeddings=128, num_conv_pos_embedding_groups=16,
    do_stable_layer_norm=True, layer_norm_eps=1e-5,
    hidden_dropout=0.1, attention_dropout=0.1, activation_dropout=0.0, feat_proj_dropout=0.1,
    layerdrop=0.0, mask_time_prob=0.0, mask_feature_prob=0.0, apply_spec_augment=False,
)


class HFSSL(nn.Module):
    """Wraps an HF Wav2Vec2/WavLM model and returns (B, C, T) features.

    layer='last'      -> last_hidden_state (paper setting for wav2vec 2.0)
    layer=<int>       -> hidden_states[layer] (0 = CNN/projection output, 24 = last)
    layer='weighted'  -> learnable softmax-weighted sum over all hidden states
                         (proposal / ablation; the WavLM SVDD models use a similar 'SEA' merge)
    """

    def __init__(self, name_or_path: str | None = 'facebook/wav2vec2-xls-r-300m',
                 layer='last', pretrained: bool = True, gradient_checkpointing: bool = False):
        super().__init__()
        from transformers import AutoModel, Wav2Vec2Config, Wav2Vec2Model
        if pretrained and name_or_path:
            self.model = AutoModel.from_pretrained(
                name_or_path, mask_time_prob=0.0, mask_feature_prob=0.0, layerdrop=0.0)
        else:  # architecture only (weights come from a checkpoint later)
            self.model = Wav2Vec2Model(Wav2Vec2Config(**XLSR_300M_CONFIG))
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.layer = layer
        self.out_dim = self.model.config.hidden_size
        n = self.model.config.num_hidden_layers + 1
        if layer == 'weighted':
            self.layer_w = nn.Parameter(torch.zeros(n))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.ndim == 3:
            wav = wav[..., 0]
        need_all = self.layer != 'last'
        o = self.model(wav, output_hidden_states=need_all)
        if self.layer == 'last':
            h = o.last_hidden_state
        elif self.layer == 'weighted':
            hs = torch.stack(o.hidden_states, 0)              # L B T C
            w = torch.softmax(self.layer_w, 0).view(-1, 1, 1, 1)
            h = (w * hs).sum(0)
        else:
            h = o.hidden_states[int(self.layer)]
        return h.transpose(1, 2)                                 # B C T
