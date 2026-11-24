import os, sys, re
import numpy as np, torch, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '_ref'))
from nes2net.backend import Nested_Res2Net_TDNN, build_backend, count_params, count_macs
from nes2net.model import SSLNes2Net, convert_official_state_dict
from nes2net.metrics import compute_eer
from nes2net.data import pad, rawboost

torch.manual_seed(0)


@pytest.mark.parametrize('wsum', [True, False])
def test_matches_official_backend(wsum):
    import official_x, official_plain
    if wsum:
        ref = official_x.Nested_Res2Net_TDNN(Nes_ratio=[8, 8], input_channel=1024, n_output_logits=2,
                                             dilation=2, pool_func='mean', SE_ratio=[1])
    else:
        ref = official_plain.Nested_Res2Net_TDNN(Nes_ratio=[8, 8], input_channel=1024, dilation=2,
                                                 pool_func='mean', SE_ratio=[1])
    mine = Nested_Res2Net_TDNN(n_output_logits=2 if wsum else 1, weighted_sum=wsum)
    mine.load_state_dict(ref.state_dict(), strict=True)
    ref.eval(); mine.eval()
    x = torch.randn(2, 1024, 201)
    assert torch.allclose(ref(x), mine(x), atol=0, rtol=0)
    assert 505_000 < count_params(mine) < 515_000       # paper: 511k


def test_astp_and_ablation_switches_run():
    x = torch.randn(2, 1024, 50)
    for kw in [dict(pool_func='ASTP'), dict(use_se=False), dict(outer_hier=False),
               dict(inner_hier=False), dict(Nes_ratio=[16, 4]), dict(Nes_ratio=[4, 8]),
               dict(weighted_sum=False, inner_hier=False), dict(kernel_size=5, dilation=1)]:
        m = Nested_Res2Net_TDNN(**kw).eval()
        assert m(x).shape == (2, 2)
    for n in ['dr_res2net', 'mean_linear']:
        assert build_backend(n)(x).shape == (2, 2)


def _fake_fairseq_keys(hf_sd):
    """Independent spec of fairseq Wav2Vec2Model key names (fairseq/models/wav2vec/wav2vec2.py)."""
    out = {}
    for k, v in hf_sd.items():
        m = re.match(r'feature_extractor\.conv_layers\.(\d+)\.conv\.(\w+)', k)
        if m: out[f'feature_extractor.conv_layers.{m[1]}.0.{m[2]}'] = v; continue
        m = re.match(r'feature_extractor\.conv_layers\.(\d+)\.layer_norm\.(\w+)', k)
        if m: out[f'feature_extractor.conv_layers.{m[1]}.2.1.{m[2]}'] = v; continue
        if k.startswith('feature_projection.layer_norm.'): out[k.replace('feature_projection.', '')] = v; continue
        if k.startswith('feature_projection.projection.'): out[k.replace('feature_projection.projection', 'post_extract_proj')] = v; continue
        if k == 'masked_spec_embed': out['mask_emb'] = v; continue
        if 'pos_conv_embed' in k:
            suf = {'weight_g': 'weight_g', 'weight_v': 'weight_v', 'bias': 'bias',
                   'parametrizations.weight.original0': 'weight_g',
                   'parametrizations.weight.original1': 'weight_v'}[k.split('conv.', 1)[1]]
            out['encoder.pos_conv.0.' + suf] = v; continue
        if k.startswith('encoder.layer_norm.'): out[k] = v; continue
        k2 = (k.replace('attention.', 'self_attn.').replace('.layer_norm.', '.self_attn_layer_norm.')
               .replace('feed_forward.intermediate_dense', 'fc1').replace('feed_forward.output_dense', 'fc2'))
        out[k2] = v
    out['quantizer.vars'] = torch.zeros(1); out['project_q.weight'] = torch.zeros(1)
    out['final_proj.weight'] = torch.zeros(1)
    return out


def test_fairseq_conversion_roundtrip():
    from transformers import Wav2Vec2Config, Wav2Vec2Model
    from nes2net.frontend import XLSR_300M_CONFIG
    cfg = dict(XLSR_300M_CONFIG, num_hidden_layers=2, hidden_size=64, intermediate_size=128,
               num_attention_heads=4, num_conv_pos_embedding_groups=4)
    model = SSLNes2Net(pretrained_ssl=False)
    model.ssl_model.model = Wav2Vec2Model(Wav2Vec2Config(**cfg))
    model.Nested_Res2Net_TDNN = build_backend('nes2net_x', input_channel=64, Nes_ratio=[4, 4])
    ref = {k: torch.randn_like(v.float()).to(v.dtype) if v.is_floating_point() else v
           for k, v in model.state_dict().items()}
    fair = {'ssl_model.model.' + k: v for k, v in
            _fake_fairseq_keys({k[16:]: v for k, v in ref.items() if k.startswith('ssl_model.model.')}).items()}
    fair.update({'module.' + k: v for k, v in ref.items() if k.startswith('Nested')})
    fair['ssl_model.model.mask_emb'] = torch.randn(64)      # present in real fairseq ckpts
    conv = convert_official_state_dict(fair, model)
    missing, unexpected = model.load_state_dict(conv, strict=False)
    assert not missing and not unexpected
    for k, v in model.state_dict().items():
        assert torch.equal(v, ref[k]), k


def test_full_model_forward_small():
    from transformers import Wav2Vec2Config, Wav2Vec2Model
    from nes2net.frontend import XLSR_300M_CONFIG
    cfg = dict(XLSR_300M_CONFIG, num_hidden_layers=2)
    m = SSLNes2Net(pretrained_ssl=False)
    m.ssl_model.model = Wav2Vec2Model(Wav2Vec2Config(**cfg))
    m.eval()
    with torch.no_grad():
        feats = m.ssl_model(torch.randn(2, 64600))
        out = m(torch.randn(2, 64600))
    assert feats.shape == (2, 1024, 201)        # 64,600 samples -> 201 frames
    assert out.shape == (2, 2)


def test_eer_and_pad():
    assert compute_eer(np.array([1., 2, 3]), np.array([-1., -2]))[0] == 0
    assert abs(compute_eer(np.array([0., 1]), np.array([0., 1]))[0] - 0.5) < 1e-9
    x = np.arange(10, dtype=np.float32)
    assert pad(x, 25).tolist() == (list(range(10)) * 3)[:25]
    assert pad(np.arange(100.), 25).tolist() == list(range(25))


def test_rawboost_algos():
    x = (np.random.randn(16000) * 0.1).astype(np.float32)
    for a in range(9):
        y = rawboost(x.copy(), a)
        assert y.shape == x.shape and np.isfinite(y).all()
