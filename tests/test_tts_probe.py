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
