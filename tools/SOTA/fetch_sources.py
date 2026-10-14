"""Fetch pinned upstream code without vendoring unlicensed repositories."""
import argparse
import json
import subprocess
from pathlib import Path


def verify(root, name):
    spec = json.loads(Path(__file__).with_name('sources.json').read_text())[name]
    path = Path(root).resolve() / name
    actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != spec['commit']:
        raise RuntimeError(f'{path}: expected {spec["commit"]}, got {actual}')
    dirty = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if dirty.strip():
        raise RuntimeError(f'{path}: tracked upstream files were modified; use a clean checkout')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='third_party/sota')
    parser.add_argument('--methods', nargs='+')
    args = parser.parse_args()
    specs = json.loads(Path(__file__).with_name('sources.json').read_text())
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in args.methods or specs:
        spec = specs[name]
        path = root / name
        if not path.exists():
            subprocess.run(['git', 'clone', '--no-checkout', spec['url'], str(path)], check=True)
            subprocess.run(['git', '-C', str(path), 'checkout', '--detach', spec['commit']], check=True)
        verify(root, name)
        print(f'{name}: {spec["commit"]}')


if __name__ == '__main__':
    main()
