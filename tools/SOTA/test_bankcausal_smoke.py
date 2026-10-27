"""Exercise the official BankCausal runner with synthetic annotations."""
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from mmengine.config import Config


def resize_pipeline(node):
    if isinstance(node, dict):
        if node.get('type') == 'Resize':
            node['scale'] = 72
        if node.get('type') == 'RandomResizedCrop':
            node['scale'] = 64
        if node.get('type') in ('CenterCrop', 'RandomCrop'):
            node['crop_size'] = 64
        for value in node.values():
            resize_pipeline(value)
    elif isinstance(node, (tuple, list)):
        for value in node:
            resize_pipeline(value)


def main():
    here = Path(__file__).resolve().parent
    source = os.environ.get('SOTA_SOURCE_ROOT', 'third_party/sota')
    with tempfile.TemporaryDirectory(prefix='bank-smoke-') as tmp:
        root = Path(tmp)
        (root/'FFA').mkdir()
        for split in ('train', 'val', 'test'):
            rows = []
            for i in range(4):
                path = root/f'FFA/{split}_{i}.png'
                Image.fromarray(np.random.default_rng(i).integers(0, 255, (80, 80, 3), dtype=np.uint8)).save(path)
                rows.append(dict(img_path=str(path), gt_score=[i % 2, i//2, 1-i % 2]))
            # The explicit image root relocation accepts project-style paths.
            for row in rows:
                row['img_path'] = 'data/FFA/'+Path(row['img_path']).name
            (root/f'{split}.pkl').write_bytes(pickle.dumps(dict(metainfo=dict(classes=['a', 'b', 'c']), data_list=rows)))
        work = root/'run'
        command = [sys.executable, '-B', str(here/'prepare_prior.py'), '--dataset', 'FFA', '--source-root', source,
                   '--image-root', str(root), '--train-ann', str(root/'train.pkl'), '--val-ann', str(root/'val.pkl'),
                   '--test-ann', str(root/'test.pkl'), '--work-dir', str(work), '--batch-size', '2', '--workers', '0']
        subprocess.run(command, check=True)
        cfg = Config.fromfile(work/'resolved.py')
        cfg.model.backbone.init_cfg = None
        cfg.model.head.img_size = 64
        cfg.model.head.hidden_dim = 32
        cfg.model.head.dim_feedforward = 64
        cfg.model.head.nheads = 4
        cfg.model.head.swav_loss.feat_dim = 32
        cfg.train_cfg.max_epochs = 1
        cfg.default_hooks.pop('ema', None)
        for hook in cfg.custom_hooks:
            if hook['type'] == 'MultiLabelSwAVHook':
                hook['feat_dim'] = 32
                hook['queue_length'] = 8
        for split in ('train', 'val', 'test'):
            resize_pipeline(cfg[f'{split}_dataloader'].dataset.pipeline)
        cfg.dump(work/'smoke.py')
        subprocess.run([sys.executable, '-B', str(here/'train_bankcausal.py'), str(work/'smoke.py'),
                        '--source-root', source], check=True,
                       env=dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MPLCONFIGDIR='/tmp/sota-mpl'))
        result = json.loads((work/'test_metrics.json').read_text())
        assert 0 <= result['mAUC'] <= 1
        checkpoint = __import__('torch').load(next(work.glob('best_*.pth')), map_location='cpu')
        # Val/test images and labels are identical in this synthetic fixture.
        # Changing inference branches at test time must fail this assertion.
        assert abs(checkpoint['message_hub']['runtime_info']['best_score'] - result['mAUC']) < 1e-6
        assert len(pickle.loads((work/'pred_results.pkl').read_bytes())) == 4
        cfg.work_dir = str(root/'evaluation')
        cfg.dump(work/'evaluation.py')
        subprocess.run([sys.executable, '-B', str(here/'train_bankcausal.py'), str(work/'evaluation.py'),
                        '--source-root', source, '--eval', str(next(work.glob('best_*.pth')))], check=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                                MPLCONFIGDIR='/tmp/sota-mpl'))
        evaluated = json.loads((root/'evaluation/test_metrics.json').read_text())
        assert abs(evaluated['mAUC'] - result['mAUC']) < 1e-6
        print('BankCausal complete training smoke passed')


if __name__ == '__main__':
    main()
