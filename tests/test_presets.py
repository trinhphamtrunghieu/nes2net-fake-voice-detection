import json, os, re, sys, ast
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytest
from nes2net import presets

ROOT = os.path.join(os.path.dirname(__file__), '..')


@pytest.mark.parametrize('name', presets.names())
@pytest.mark.parametrize('plat', ['kaggle', 'colab', 'local'])
def test_presets_resolve(name, plat):
    cfg = presets.get(name, plat)
    assert cfg['preset'] == name and set(cfg['run']) >= {'A', 'A_full', 'B', 'B_eval', 'C', 'L'}
    from nes2net.ablation import ABLATIONS
    assert set(cfg['C']['groups']) <= set(ABLATIONS)
    from nes2net.datasets_zoo import get_items  # noqa: F401  (specs are parsed lazily; just import)


def test_paper_recipe_and_budgets():
    c = presets.get('paper_train', 'kaggle')
    assert (c['B']['lr'], c['B']['bs'], c['B']['epochs'], c['B']['algo'], c['B']['dev']) == \
           (2.5e-7, 12, 100, 4, 'asv19:validation')
    assert c['B']['time_budget_h'] == 11.0 and presets.get('paper_train', 'colab')['B']['time_budget_h'] == 11.5
    assert presets.get('paper_train', 'local')['B']['time_budget_h'] is None


def test_overrides_deep_merge_and_isolation():
    c = presets.get('default', 'colab', {'B': {'lr': 3e-6}, 'run': {'L': False}})
    assert c['B']['lr'] == 3e-6 and c['B']['bs'] == 12 and c['run']['L'] is False and c['run']['A']
    assert presets.get('default', 'colab')['B']['lr'] == 1e-6      # base not mutated
    with pytest.raises(ValueError):
        presets.get('nope')


def test_single_notebook_is_valid_and_patchable():
    p = os.path.join(ROOT, 'notebooks', 'nes2net.ipynb')
    nb = json.load(open(p))
    srcs = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    for s in srcs:
        ast.parse(s)
    assert sum(len(re.findall(r"^PRESET = '[a-z_]+'", s, re.M)) for s in srcs) == 1
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    from kaggle_push import notebook_with_preset
    out = notebook_with_preset('smoke', ref='abc1234')
    assert "PRESET = 'smoke'" in out and "REPO_REF = 'abc1234'" in out


def test_run_module_imports_and_cli_overrides(monkeypatch):
    from nes2net import run
    for fn in ('sanity', 'eval_official', 'eval_official_full', 'train', 'eval_trained',
               'ablation', 'layer_study', 'run_all', 'results_table'):
        assert callable(getattr(run, fn))
    t = run.results_table({'X': {'n': 10, 'eer': 1.234, 'minDCF': 0.1},
                           'ASVspoof2021 DF': {'phase_eval': {'n': 5, 'eer': 2.0, 'minDCF': 0.2},
                                               'all_trials': {'n': 9, 'eer': 3.0, 'minDCF': 0.3}}})
    assert list(t['subset']) == ['', 'phase_eval', 'all_trials'] and 'paper 1.65' in t['reference'].iloc[1]
