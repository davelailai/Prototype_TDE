"""Verify official shared-class sampling, including normal anchors and warmup."""
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from data import Images


class PairingTest(unittest.TestCase):
    def test_shared_class_and_normal_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'FFA').mkdir()
            labels = [[1, 0], [1, 1], [0, 1], [0, 0], [0, 0]]
            rows = []
            for index, label in enumerate(labels):
                Image.new('RGB', (40, 40), (index*40, 20, 10)).save(root/f'FFA/{index}.png')
                rows.append(dict(img_path=f'data/FFA/{index}.png', gt_score=label))
            annotation = root/'annotations.pkl'
            annotation.write_bytes(pickle.dumps(dict(metainfo=dict(classes=['a', 'b']), data_list=rows)))
            dataset = Images(annotation, root, size=32, train=True, method='CIPL')
            for index in range(len(dataset)):
                for _ in range(20):
                    views, pair, _ = dataset[index]
                    normal = pair.sum(-1, keepdim=True) == 0
                    augmented = np.concatenate([pair.numpy(), normal.numpy()], axis=1)
                    self.assertTrue((augmented[0]*augmented[1]).any())
                    self.assertEqual(len(views), 4)
                    self.assertEqual(tuple(views[0].shape), (3, 32, 32))
            warmup = Images(annotation, root, size=32, train=True, method='CIPL', cipl_pairing=False)
            image, label, _ = warmup[0]
            self.assertEqual(tuple(image.shape), (3, 32, 32))
            self.assertEqual(tuple(label.shape), (2,))


if __name__ == '__main__':
    unittest.main()
