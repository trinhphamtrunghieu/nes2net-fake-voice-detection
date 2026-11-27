import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nes2net.tts_probe import build_probe, _rms_norm, TARGET_RMS
from nes2net import engine


def test_probe_pairs_and_normalisation(tmp_path):
    rng = np.random.default_rng(0)
    rows = [{'id': str(i), 'text': f'câu số {i}', 'gender': ['male', 'female'][i % 2],
             'audio': (rng.standard_normal(16000 + 400 * i) * (0.01 + 0.02 * i)).astype(np.float32)} for i in range(6)]
    calls = []
    def fake_tts(text):
        calls.append(text)
        return None if text.endswith('5') else (np.sin(np.arange(20000) / 7) * 0.3).astype(np.float32)
    P = build_probe('vi_vn', 'facebook/mms-tts-vie', str(tmp_path), rows=rows, synth=fake_tts, log=lambda *a: None)
    raw, norm = P['raw'], P['norm']
    assert len(raw) == len(norm) == 10                  # 5 renderable sentences x (bona + tts)
    assert sum(it.label for it in raw) == 5 and {it.meta['gender'] for it in raw} == {'male', 'female'}
    # cached: a second build does not call the TTS again
    n = len(calls); build_probe('vi_vn', 'facebook/mms-tts-vie', str(tmp_path), rows=rows, synth=fake_tts, log=lambda *a: None)
    assert len(calls) == n + 1                          # only the unrenderable one is retried
    from nes2net.data import load_audio
    for it in norm:
        x = load_audio(it.source)
        assert abs(np.sqrt(np.mean(x ** 2)) - TARGET_RMS) < 0.01 or np.max(np.abs(x)) > 0.98
    # perfect detector on this probe -> EER 0
    s = {it.key: float(it.label) for it in raw}
    assert engine.evaluate_scores(s, raw, by_attack=False)['eer'] == 0


def test_run_vi_zeroshot_with_fakes(tmp_path, monkeypatch):
    from nes2net import run, presets, tts_probe
    from nes2net.env import Env
    from tests.test_pipeline import _tiny_model
    rng = np.random.default_rng(1)
    rows = [{'id': str(i), 'text': f's{i}', 'gender': ['male', 'female'][i % 2],
             'audio': (rng.standard_normal(20000) * 0.05).astype(np.float32)} for i in range(4)]
    orig = tts_probe.build_probe
    fake_build = lambda lang, tts, cache, **kw: orig(
        lang, tts, cache, rows=rows, synth=lambda t: (rng.standard_normal(18000) * 0.1).astype(np.float32),
        log=lambda *a: None)
    monkeypatch.setattr(tts_probe, 'build_probe', fake_build)
    monkeypatch.setattr(run, '_official_model', lambda cfg, env: _tiny_model().eval())
    env = Env('local', str(tmp_path / 'w'), str(tmp_path / 't'), 0, 'cpu')
    os.makedirs(env.work); os.makedirs(env.tmp)
    res = run.vi_zeroshot(presets.get('vi_zeroshot', 'local'), env)
    assert len(res) == 2 * 2 * 2                         # 2 probes x raw/norm x 4s/full
    t = run.vi_table(res)
    assert set(t['window']) == {'4s', 'full utt'} and 'eer_bona_male' in next(iter(res.values()))
    assert os.path.exists(os.path.join(env.work, 'partVI_results.json'))


def test_diag_data_with_fakes(tmp_path, monkeypatch):
    import io, soundfile as sf
    from nes2net import run, presets, datasets_zoo
    from nes2net.data import Item
    from nes2net.env import Env
    from tests.test_pipeline import _tiny_model
    rng = np.random.default_rng(2)
    def wav(n, amp):
        b = io.BytesIO(); sf.write(b, (rng.standard_normal(n) * amp).astype(np.float32), 16000, format='FLAC'); return b.getvalue()
    def fake_items(spec, cache_dir=None):
        return [Item(f'{spec}_{i}', wav(16000 + 100 * i, 0.05), int(i % 3 == 0),
                     {'attack': '-' if i % 3 == 0 else 'A0%d' % (i % 3)}) for i in range(12)]
    monkeypatch.setattr(datasets_zoo, 'get_items', fake_items)
    monkeypatch.setattr(run, '_official_model', lambda cfg, env: _tiny_model().eval())
    env = Env('local', str(tmp_path / 'w'), str(tmp_path / 't'), 0, 'cpu')
    os.makedirs(env.work); os.makedirs(env.tmp)
    out = run.diag_data(presets.get('diag_data', 'local', {'DIAG': {'n': 8}}), env)
    assert set(out['official_eer']) == {'train', 'dev', 'test'}
    assert out['crosstab']['train']['1'] == {'-': 4} and out['stats']['dev']['1']['formats'] == {'FLAC/PCM_16': 4}
    eer, st, ct = run.diag_tables(out)
    assert len(eer) == 3 and len(st) == 6 and len(ct) == 6


def test_diag_train_with_fakes(tmp_path, monkeypatch):
    import io, soundfile as sf, huggingface_hub
    from nes2net import run, presets, datasets_zoo, model as M
    from nes2net.data import Item
    from nes2net.env import Env
    from tests.test_pipeline import _tiny_model
    rng = np.random.default_rng(3)
    def wav(n):
        b = io.BytesIO(); sf.write(b, (rng.standard_normal(n) * 0.05).astype(np.float32), 16000, format='FLAC'); return b.getvalue()
    pool = {s: [Item(f'{s}_{i}', wav(8000), int(i % 3 == 0)) for i in range(18)] for s in ('tr', 'dv')}
    monkeypatch.setattr(datasets_zoo, 'get_items', lambda spec, cache_dir=None: pool['tr' if 'train' in spec else 'dv'])
    monkeypatch.setattr(run, '_build_train_model', lambda cfg, device='cuda': _tiny_model())
    monkeypatch.setattr(M, 'load_official_checkpoint', lambda path, **kw: _tiny_model().eval())
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', lambda *a, **k: 'unused')
    monkeypatch.setattr(run, '_pos_conv_check', lambda m, name: {'loaded_ok': True})
    env = Env('local', str(tmp_path / 'w'), str(tmp_path / 't'), 0, 'cpu')
    os.makedirs(env.work); os.makedirs(env.tmp)
    cfg = presets.get('diag_train', 'local', {'DT': {'n_train': 12, 'n_dev': 6, 'n_score': 6, 'bs': 4,
                                                     'variants': [['ft', 'official', 1, {'amp': False}],
                                                                  ['x', 'xlsr', 1, {'amp': False}]]}})
    out = run.diag_train(cfg, env)
    assert set(out['official']) == {f'{s}/{m}' for s in ('train', 'dev') for m in ('eval', 'bn_batch', 'train')}
    assert set(out['variants']) == {'ft', 'x'} and len(out['variants']['x']['history']) == 1
    assert not any(f.endswith('.pth') for _, _, fs in os.walk(env.work) for f in fs)
    t = run.diag_train_table(out)
    assert list(t['model']) == ['official ckpt', 'ft', 'x']


def test_bn_batch_mode_keeps_running_stats():
    from nes2net import run
    from nes2net.backend import build_backend
    m = build_backend('nes2net_x', input_channel=64).eval()
    bn = next(x for x in m.modules() if isinstance(x, torch.nn.BatchNorm1d))
    before = bn.running_mean.clone()
    run._set_mode(m, 'bn_batch')
    assert bn.training and not m.training
    with torch.no_grad():
        m(torch.randn(4, 64, 30) * 5 + 3)
    assert torch.equal(bn.running_mean, before)
    run._set_mode(m, 'eval')
    assert not bn.training and bn.momentum == 0.1


def test_diag_ckpt_with_fakes(tmp_path, monkeypatch):
    import io, soundfile as sf
    from nes2net import run, presets, datasets_zoo, model as M
    from nes2net.data import Item
    from nes2net.env import Env
    from tests.test_pipeline import _tiny_model
    rng = np.random.default_rng(4)
    def wav():
        b = io.BytesIO(); sf.write(b, (rng.standard_normal(8000) * 0.05).astype(np.float32), 16000, format='FLAC'); return b.getvalue()
    pool = [Item(f'u{i}', wav(), int(i % 2)) for i in range(12)]
    monkeypatch.setattr(datasets_zoo, 'get_items', lambda spec, cache_dir=None: pool)
    tiny = _tiny_model()
    monkeypatch.setattr(M, 'SSLNes2Net', lambda **kw: _tiny_model())
    env = Env('local', str(tmp_path / 'w'), str(tmp_path / 't'), 0, 'cpu')
    os.makedirs(env.work + '/runB'); os.makedirs(env.tmp)
    torch.save({'model': tiny.state_dict(), 'epoch': 3}, env.work + '/runB/last_state.pth')
    out = run.diag_ckpt(presets.get('diag_ckpt', 'local', {'DC': {'path': 'runB/last_state.pth', 'n': 8}}), env)
    assert set(out['eer']) == {f'{s}/{m}' for s in ('train', 'dev') for m in ('eval', 'bn_batch', 'train')}


def test_ssl_dropout_matches_fairseq_by_default():
    from nes2net.model import SSLNes2Net
    m = SSLNes2Net(pretrained_ssl=False)
    ps = [x.p for x in m.ssl_model.model.modules() if isinstance(x, torch.nn.Dropout)]
    att = [x.dropout for n, x in m.ssl_model.model.named_modules() if n.endswith('attention')]
    assert ps and all(p == 0 for p in ps) and att and all(a == 0 for a in att)
    m2 = SSLNes2Net(pretrained_ssl=False, ssl_dropout='hf')
    assert max(x.p for x in m2.ssl_model.model.modules() if isinstance(x, torch.nn.Dropout)) == 0.1


def test_dev_aug_cache_is_deterministic_and_reused(tmp_path):
    import io, soundfile as sf
    from nes2net.data import Item
    from nes2net import engine
    rng = np.random.default_rng(5)
    def wav(n):
        b = io.BytesIO(); sf.write(b, (rng.standard_normal(n) * 0.05).astype(np.float32), 16000, format='FLAC'); return b.getvalue()
    items = [Item(f'd{i}', wav(30000 + 1000 * i), int(i % 2)) for i in range(10)]
    d = str(tmp_path / 'cache')
    engine.build_aug_cache(items, d, algo=4, seed=7, workers=0, log=lambda *a: None)
    ds = engine.CachedWaves(d)
    assert len(ds) == 10 and ds.keys == [it.key for it in items] and ds.labels == [it.label for it in items]
    ref = engine._SeededAug(items, 4, 7)
    x0 = ds[3][0]
    assert torch.equal(x0, ref[3][0])                       # same seed -> same augmented wave
    assert not torch.allclose(x0, engine.WaveDataset(items[3:4], algo=0)[0][0])   # it is augmented
    m = os.path.getmtime(os.path.join(d, 'waves.npy'))
    engine.build_aug_cache(items, d, algo=4, seed=7, workers=0, log=lambda *a: None)
    assert os.path.getmtime(os.path.join(d, 'waves.npy')) == m   # reused, not rebuilt
    engine.build_aug_cache(items[:5], d, algo=4, seed=7, workers=0, log=lambda *a: None)
    assert len(engine.CachedWaves(d)) == 5                 # different items -> rebuilt


def test_train_with_val_aug_cache(tmp_path):
    import io, soundfile as sf
    from nes2net.data import Item
    from nes2net import engine
    from tests.test_pipeline import _tiny_model
    rng = np.random.default_rng(6)
    def wav():
        b = io.BytesIO(); sf.write(b, (rng.standard_normal(9000) * 0.05).astype(np.float32), 16000, format='FLAC'); return b.getvalue()
    items = [Item(f'u{i}', wav(), int(i % 2)) for i in range(8)]
    hist = engine.train(_tiny_model(), items, items, str(tmp_path / 'run'), epochs=1, bs=4, algo=4, device='cpu',
                        amp=False, workers=0, tb=False, val_aug_cache=str(tmp_path / 'cache'))
    assert len(hist) == 1 and np.isfinite(hist[0]['val_loss_w_aug'])
    assert os.path.exists(tmp_path / 'cache' / 'DONE')
