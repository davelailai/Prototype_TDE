"""Run a complete tiny train/val/test cycle, without GPUs or downloads."""
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix='sota-smoke-') as tmp:
        root = Path(tmp)
        (root/'FFA').mkdir()
        for split in ('train', 'val', 'test'):
            rows = []
            for index in range(4):
                relative = f'FFA/{split}_{index}.png'
                image = np.random.default_rng(index).integers(0, 256, (80, 80, 3), dtype=np.uint8)
                Image.fromarray(image).save(root/relative)
                rows.append(dict(img_path='data/'+relative, gt_score=[index % 2, (index//2) % 2, 1-index % 2],
                                 group_id=f'{split}_{index}'))
            (root/f'{split}.pkl').write_bytes(pickle.dumps(dict(metainfo=dict(classes=['a', 'b', 'c']), data_list=rows)))
        for method in os.environ.get('SOTA_SMOKE_METHODS', 'MLC-PAT,CIPL,ProCI,MICCAI2024').split(','):
            work = root/method
            command = [sys.executable, '-B', str(here/'train.py'), '--method', method, '--dataset', 'FFA',
                       '--image-root', str(root), '--train-ann', str(root/'train.pkl'),
                       '--val-ann', str(root/'val.pkl'), '--test-ann', str(root/'test.pkl'),
                       '--source-root', os.environ.get('SOTA_SOURCE_ROOT', 'third_party/sota'),
                       '--work-dir', str(work), '--device', 'cpu', '--scratch', '--epochs', '1',
                       '--image-size', '64', '--workers', '0', '--batch-size', '2']
            subprocess.run(command, check=True, env=dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'))
            result = json.loads((work/'test_metrics.json').read_text())
            assert 0 <= result['mAUC'] <= 1
            predictions = pickle.loads((work/'pred_results.pkl').read_bytes())
            assert len(predictions) == 4
            assert predictions[0]['group_id'] == 'test_0'
            # Check the same checkpoint can be evaluated without pretraining downloads.
            subprocess.run(command + ['--eval', str(work/'best.pth')], check=True,
                           env=dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'))
            print(f'{method}: end-to-end smoke passed', flush=True)


if __name__ == '__main__':
    main()
