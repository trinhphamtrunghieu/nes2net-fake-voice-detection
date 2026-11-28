"""CPU smoke test of the whole pipeline with a tiny SSL model and synthetic audio."""
import io, json, os, sys
import numpy as np, soundfile as sf, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nes2net.data import hf_items, WaveDataset
from nes2net import engine, ablation
from nes2net.model import SSLNes2Net


def _wav_bytes(n, seed):
    r = np.random.default_rng(seed); buf = io.BytesIO()
    sf.write(buf, (r.standard_normal(n) * 0.05).astype(np.float32), 16000, format='FLAC')
    return buf.getvalue()


def _fake_bench(tmp, n=16):
    import pyarrow as pa, pyarrow.parquet as pq
    t = pa.table({'path': [f'u{i}.flac' for i in range(n)],
                  'audio': [{'bytes': _wav_bytes(8000 + 997 * i, i), 'path': None} for i in range(n)],
                  'label': [i % 2 for i in range(n)],
                  'notes': [json.dumps({'utterance_id': f'U{i}', 'subset': 'eval' if i < 12 else 'progress',
                                        'attack_id': 'A0%d' % (i % 3)}) for i in range(n)]})
    pq.write_table(t, os.path.join(tmp, 'test-00000.parquet'))
    return os.path.join(tmp, 'test-00000.parquet')


def _tiny_model():
    from transformers import Wav2Vec2Config, Wav2Vec2Model
    from nes2net.frontend import XLSR_300M_CONFIG
    m = SSLNes2Net(pretrained_ssl=False)
    m.ssl_model.model = Wav2Vec2Model(Wav2Vec2Config(**dict(XLSR_300M_CONFIG, num_hidden_layers=2)))
    return m


def test_end_to_end(tmp_path):
    from datasets import load_dataset
    p = _fake_bench(str(tmp_path))
    ds = load_dataset('parquet', data_files=[p], split='train')
    keep = lambda m: m.get('subset') == 'eval'
    meta = lambda r: {**json.loads(r['notes']), 'attack': json.loads(r['notes'])['attack_id']}
    items = hf_items(ds, lambda r: int(r['label'] == 0), lambda r, j: json.loads(r['notes'])['utterance_id'],
                     meta, keep=keep)
    assert len(items) == 12 and items[0].key == 'U0' and items[0].label == 1 and items[1].label == 0
    x, y, k = WaveDataset(items, algo=4)[3]
    assert x.shape == (64600,)
    m = _tiny_model()
    hist = engine.train(m, items, items, str(tmp_path / 'run'), epochs=2, bs=4, algo=4, device='cpu',
                        amp=False, workers=0)
    best = engine.best_epochs(hist, k=2)
    engine.average_checkpoints(m, [str(tmp_path / 'run' / f'epoch_{e}.pth') for e in best])
    s = engine.score(m, items, bs=4, device='cpu', workers=0, amp=False)
    r = engine.evaluate_scores(s, items)
    assert r['n'] == 12 and 0 <= r['eer'] <= 100 and set(r['per_attack_eer']) <= {'A00', 'A01', 'A02'}
    s_full = engine.score(m, items[:3], device='cpu', workers=0, amp=False, full=True)
    assert len(s_full) == 3
    # frozen-feature ablation path
    d = ablation.extract_features(m.ssl_model.model, items, str(tmp_path / 'feat'), layers=(1, 2),
                                  bs=4, device='cpu', workers=0, amp=False)
    res = ablation.train_backend({'backend': 'nes2net_x', 'Nes_ratio': [4, 4]}, d, d, {'ev': d},
                                 layers=(1, 2), epochs=2, bs=4, device='cpu', workers=0)
    assert 'ev_eer' in res and res['params_k'] > 0


def test_resume_and_prune(tmp_path):
    from datasets import load_dataset
    p = _fake_bench(str(tmp_path))
    ds = load_dataset('parquet', data_files=[p], split='train')
    items = hf_items(ds, lambda r: int(r['label'] == 0), lambda r, j: f'k{j}')
    out = str(tmp_path / 'run')
    logs = []
    kw = dict(bs=4, algo=0, device='cpu', amp=False, workers=0, keep_best_k=1, log=logs.append, log_every=1)
    # 1) stop mid-epoch through the time budget -> state saved with step > 0
    h = engine.train(_tiny_model(), items, items, out, epochs=3, time_budget_h=1e-9, **kw)
    st = torch.load(os.path.join(out, 'last_state.pth'), weights_only=False)
    assert h == [] and st['epoch'] == 1 and st['step'] == 1 and 'shuffle_seed' in st
    # 2) resume: finishes epoch 1 (remaining steps) then epochs 2-3
    h = engine.train(_tiny_model(), items, items, out, epochs=3, **kw)
    assert [x['epoch'] for x in h] == [1, 2, 3]
    assert any('resumed' in l for l in logs)
    # 3) already finished -> nothing more to do
    h2 = engine.train(_tiny_model(), items, items, out, epochs=3, **kw)
    assert len(h2) == 3
    # logging: text log + JSONL metrics with step/epoch/resume/stop events
    from nes2net.logutil import load_metrics
    df = load_metrics(os.path.join(out, 'metrics.jsonl'))
    assert {'start', 'train_step', 'epoch', 'resume', 'stop', 'finished'} <= set(df.event)
    ep = df[df.event == 'epoch']
    assert ep['epoch'].tolist() == [1, 2, 3] and ep['val_eer_wo_aug'].notna().all()
    assert 'utt_per_s' in df[df.event == 'train_step'].columns
    assert 'resumed' in open(os.path.join(out, 'train.log')).read()
    kept = [f for f in os.listdir(out) if f.startswith('epoch_')]
    assert 1 <= len(kept) <= 2               # keep_best_k=1 for each of the two val criteria


def test_full_eval_shardwise(tmp_path, monkeypatch):
    import shutil
    from nes2net import full_eval, datasets_zoo
    src = _fake_bench(str(tmp_path))
    monkeypatch.setattr(full_eval, 'hfbench_files', lambda repo: ['data/test-00000.parquet', 'data/test-00001.parquet'])
    import huggingface_hub
    def fake_dl(repo, file, repo_type=None, local_dir=None):
        os.makedirs(local_dir, exist_ok=True); dst = os.path.join(local_dir, os.path.basename(file))
        shutil.copy(src, dst); return dst
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', fake_dl)
    m = _tiny_model().eval()
    out = str(tmp_path / 'fe')
    res = full_eval.score_hfbench_full(m, 'X/Fake', out, device='cpu', bs=4, workers=0,
                                       shard_dir=str(tmp_path / 'sh'))
    assert res['all_trials']['n'] == 16 and res['phase_eval']['n'] == 12   # keys repeat across the 2 fake shards
    assert not os.listdir(tmp_path / 'sh')                                  # shards deleted after scoring
    res2 = full_eval.score_hfbench_full(m, 'X/Fake', out, device='cpu', bs=4, workers=0,
                                        shard_dir=str(tmp_path / 'sh'))      # resumable: nothing rescored
    assert res2['all_trials']['n'] == 16


def test_epoch_sampler_exact_resume():
    from torch.utils.data import DataLoader
    s = engine.EpochSampler(23, seed=7)
    full = s.order(3)
    assert sorted(full) == list(range(23)) and full == engine.EpochSampler(23, 7).order(3)
    assert full != s.order(4)                                     # reshuffled every epoch
    s.set_epoch(3, skip=8)                                        # resumed after 2 batches of 4
    assert list(s) == full[8:] and len(s) == 15
    # through a DataLoader with drop_last: the resumed batches are exactly the remaining ones
    dl = DataLoader(list(range(23)), batch_size=4, sampler=s, drop_last=True)
    got = [i for b in dl for i in b.tolist()]
    assert got == full[8:8 + 12]
    s.set_epoch(4)
    assert list(s) == s.order(4)
