"""Exercise all corrected CIPL stages on synthetic data without GPU/downloads."""
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image


def main():
    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix='cipl-official-smoke-') as temporary:
        root = Path(temporary)
        (root/'FFA').mkdir()
        for split_index, split in enumerate(['train', 'val', 'test']):
            rows = []
            for index in range(8 if split == 'train' else 4):
                name = f'FFA/{split}_{index}.png'
                image = np.random.default_rng(split_index*100+index).integers(0, 256, (80, 80, 3), dtype=np.uint8)
                Image.fromarray(image).save(root/name)
                rows.append(dict(img_path='data/'+name, gt_score=[index % 2, (index//2) % 2],
                                 group_id=f'{split}_{index}'))
            (root/f'{split}.pkl').write_bytes(pickle.dumps(dict(metainfo=dict(classes=['a', 'b']), data_list=rows)))
        settings = json.loads((here.parents[1]/'configs/SOTA/methods.json').read_text())
        settings['CIPL'].update(epochs=2, warm_epochs=1, push_start=1, push_interval=1,
                                last_epochs=1, prototypes_per_class=1, prototype_dim=8)
        (root/'settings.json').write_text(json.dumps(settings))
        work = root/'run'
        command = [sys.executable, '-B', str(here/'train.py'), '--method', 'CIPL', '--dataset', 'FFA',
                   '--image-root', str(root), '--settings', str(root/'settings.json'),
                   '--work-dir', str(work), '--device', 'cpu', '--scratch', '--lr', '.005',
                   '--image-size', '64', '--workers', '0', '--batch-size', '2']
        for split in ['train', 'val', 'test']:
            command += [f'--{split}-ann', str(root/f'{split}.pkl')]
        env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        subprocess.run(command, check=True, env=env)
        record = json.loads((work/'run.json').read_text())
        assert all(record['settings'][key] == .005 for key in ['lr', 'backbone_lr', 'last_lr'])
        assert record['settings']['cipl_recipe'] == 'official_mechanisms_v2'
        history = [json.loads(line) for line in (work/'history.jsonl').read_text().splitlines()]
        assert {row['stage'] for row in history} == {'joint', 'push', 'last-0'}
        projection = json.loads((work/'projection_epoch1.json').read_text())
        assert projection['projected'] == 3
        assert len(set(projection['image_to_prototype'].values())) == 3
        result = json.loads((work/'test_metrics.json').read_text())
        assert 0 <= result['mAUC'] <= 1
        subprocess.run(command+['--eval', str(work/'best.pth')], check=True, env=env)
        print('CIPL warmup/joint/push/refinement/test/checkpoint smoke passed; uniform base LR verified.')


if __name__ == '__main__':
    main()
