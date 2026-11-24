#!/usr/bin/env python
"""Push notebooks/nes2net.ipynb to Kaggle as a kernel with a chosen preset (uses the kaggle CLI).

  python scripts/kaggle_push.py --user <kaggle-user> --preset smoke
  python scripts/kaggle_push.py --user <kaggle-user> --preset paper_train --resume   # next session

The repo keeps ONE notebook; this script only rewrites the `PRESET = ...` line in a temp copy and
writes kernel-metadata.json (private, GPU T4, Internet on). --resume adds the kernel's own previous
output as an input, so run.train() restores runB/ and continues.
"""
import argparse, json, os, re, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def notebook_with_preset(preset: str, ref: str | None = None, path=os.path.join(ROOT, 'notebooks', 'nes2net.ipynb')) -> str:
    text = open(path).read()
    new, n = re.subn(r"PRESET = '[a-z_]+'", f"PRESET = '{preset}'", text, count=1)
    assert n == 1, 'PRESET line not found in notebook'
    if ref:
        new, n = re.subn(r"REPO_REF = '[^']+'", f"REPO_REF = '{ref}'", new, count=1)
        assert n == 1
    json.loads(new)
    return new


def main():
    from nes2net import presets
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--user', required=True); p.add_argument('--preset', required=True, choices=presets.names())
    p.add_argument('--slug', help='default: nes2net-<preset>'); p.add_argument('--ref', help='pin a git commit/tag')
    p.add_argument('--resume', action='store_true'); p.add_argument('--accelerator', default='NvidiaTeslaT4')
    a = p.parse_args()
    slug = a.slug or 'nes2net-' + a.preset.replace('_', '-')
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, 'nes2net.ipynb'), 'w').write(notebook_with_preset(a.preset, a.ref))
        meta = {'id': f'{a.user}/{slug}', 'title': slug, 'code_file': 'nes2net.ipynb', 'language': 'python',
                'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
                'enable_internet': True, 'dataset_sources': [], 'competition_sources': [],
                'kernel_sources': [f'{a.user}/{slug}'] if a.resume else [], 'model_sources': []}
        json.dump(meta, open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=2)
        subprocess.run(['kaggle', 'kernels', 'push', '-p', d, '--accelerator', a.accelerator], check=True)
    print(f'https://www.kaggle.com/code/{a.user}/{slug}')


if __name__ == '__main__':
    main()
