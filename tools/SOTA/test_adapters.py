"""CPU tests. Set SOTA_SOURCE_ROOT to the pinned upstream checkouts."""
import copy
import io
import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image

from adapters import CIPL, PAT, ProCI
from common import append_normal, asymmetric_loss, metrics, paired_ce, patch_logits
from data import Images, resolve_image
from train import update_ema
from protocol import ReferenceLoss, reference_protocol

torch.set_num_threads(2)
SOURCE = Path(os.environ.get('SOTA_SOURCE_ROOT', 'third_party/sota'))
SETTINGS = json.loads((Path(__file__).resolve().parents[2]/'configs/SOTA/methods.json').read_text())


class ResToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)

    def forward(self, images):
        return self.conv(images)


class HelpersTest(unittest.TestCase):
    def test_patch_major_and_permutation(self):
        a = torch.randn(8, 3)
        v = torch.randn(8, 3)
        output = patch_logits(a, v, 2)
        expected = (a.reshape(4, 2, 3).softmax(0)*v.reshape(4, 2, 3)).sum(0)
        torch.testing.assert_close(output, expected)
        order = torch.tensor([2, 0, 3, 1])
        torch.testing.assert_close(output, patch_logits(a.reshape(4, 2, 3)[order].reshape(8, 3),
                                                       v.reshape(4, 2, 3)[order].reshape(8, 3), 2))

    def test_all_diseases_in_cipl_loss(self):
        logits = torch.zeros(2, 4, requires_grad=True)
        y = append_normal(torch.tensor([[1., 0., 0.], [0., 1., 1.]]))
        paired_ce(logits, y).backward()
        self.assertTrue((logits.grad[:, 2].abs() > 0).all())
        self.assertEqual(y.shape, (2, 4))

    def test_asl_extremes(self):
        x = torch.tensor([[-100., 100., 0.]], requires_grad=True)
        loss = asymmetric_loss(x, torch.tensor([[1., 0., 1.]]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_metrics_missing_class(self):
        result = metrics(np.array([[0, 0], [1, 0]]), np.array([[.1, .2], [.8, .1]]))
        self.assertEqual(result['mAUC'], 1.)
        self.assertIsNone(result['per_class_AUC'][1])

    def test_path_relocation_and_dataset_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'FFA').mkdir()
            Image.new('RGB', (64, 64), 'white').save(root/'FFA/image.png')
            ann = root/'ann.pkl'
            ann.write_bytes(pickle.dumps(dict(metainfo=dict(classes=['a', 'b']), data_list=[
                dict(img_path='/obsolete/project/data/FFA/image.png', gt_score=[1, 0])])) )
            self.assertEqual(resolve_image('/obsolete/project/data/FFA/image.png', root), root/'FFA/image.png')
            for method in ('CIPL', 'MLC-PAT', 'ProCI'):
                ds = Images(ann, root, 32, train=True, method=method)
                images, label, _ = ds[0]
                if method == 'MLC-PAT':
                    self.assertEqual(len(images), 5)
                if method == 'CIPL':
                    self.assertEqual(label.shape, (2, 2))
                    self.assertEqual(len(images), 4)

    def test_pat_backward_and_ema(self):
        model = PAT(ResToy(), 8, 3)
        image = [torch.randn(2, 3, 4, 4) for _ in range(5)]
        y = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
        model.loss(image, y, 0).backward()
        for head in model.heads:
            self.assertTrue(head.weight.grad.abs().sum() > 0)
        self.assertEqual(model(image).shape, (2, 3))
        ema = copy.deepcopy(model)
        with torch.no_grad():
            model.heads[0].weight.add_(1)
        old = ema.heads[0].weight.clone()
        update_ema(ema, model, .9)
        torch.testing.assert_close(ema.heads[0].weight, old+.1)


@unittest.skipUnless((SOURCE/'CIPL/model.py').exists(), 'Fetch official CIPL source')
class CIPLTest(unittest.TestCase):
    def test_joint_warm_and_checkpoint(self):
        settings = dict(SETTINGS['CIPL'], prototype_dim=4, prototypes_per_class=2)
        model = CIPL(ResToy(), 3, SOURCE/'CIPL', settings, 32)
        views = [torch.randn(2, 3, 4, 4) for _ in range(4)]
        y = torch.tensor([[[1., 0., 0.], [0., 1., 0.]], [[0., 0., 0.], [0., 0., 1.]]])
        for epoch in (0, 16):
            model.zero_grad()
            loss = model.loss(views, y, epoch)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(model.core.prototype_vectors.grad.abs().sum() > 0)
            self.assertIsNone(model.core.prototype_vectors_ema.grad)
        model.eval()
        before = model(views[0])
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        other = CIPL(ResToy(), 3, SOURCE/'CIPL', settings, 32)
        other.load_state_dict(torch.load(stream))
        other.eval()
        torch.testing.assert_close(before, other(views[0]))
        self.assertTrue(other.core._ema_param_initialized)

    def test_projection_uses_only_present_class(self):
        settings = dict(SETTINGS['CIPL'], prototype_dim=4, prototypes_per_class=2)
        model = CIPL(ResToy(), 3, SOURCE/'CIPL', settings, 32)
        old = model.core.prototype_vectors.detach().clone()
        batch = [(torch.randn(2, 3, 4, 4), torch.tensor([[1., 0., 0.]]*2), torch.arange(2))]
        model.push(batch, torch.device('cpu'))
        self.assertFalse(torch.equal(old[:2], model.core.prototype_vectors[:2]))
        torch.testing.assert_close(old[2:], model.core.prototype_vectors[2:])


@unittest.skipUnless((SOURCE/'ProCI/causal.py').exists(), 'Fetch official ProCI source')
class ProCITest(unittest.TestCase):
    def test_aligned_objective_keeps_auxiliary_scale(self):
        native = ProCI(ResToy(), 8, ['a', 'b', 'c'], SOURCE/'ProCI', SETTINGS['ProCI'])
        aligned = copy.deepcopy(native)
        aligned.classification_loss = ReferenceLoss(reference_protocol('FFA'))
        aligned.aligned_loss = True
        images = torch.randn(3, 3, 4, 4)
        labels = torch.tensor([[1., 0., 1.], [1., 1., 0.], [0., 0., 0.]])
        native_loss = native.loss(images, labels, 30)
        aligned_loss = aligned.loss(images, labels, 30)
        torch.testing.assert_close(aligned_loss, native_loss/len(images))

    def test_memory_backward_and_label_free_inference(self):
        model = ProCI(ResToy(), 8, ['a', 'b', 'c'], SOURCE/'ProCI', SETTINGS['ProCI'])
        with torch.no_grad():
            model.core.fc1.bias.fill_(10.)
        images = torch.randn(3, 3, 4, 4)
        y = torch.tensor([[1., 0., 1.], [1., 1., 0.], [0., 0., 0.]])
        for epoch in (0, 15, 30):
            model.zero_grad()
            loss = model.loss(images, y, epoch)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
        snapshot = model.core.memory.protos.detach().clone()
        counts = model.core.memory.counts.clone()
        model.zero_grad()
        smooth = model.loss(images, y, 30, write_memory=False, memory_snapshot=snapshot)
        smooth.backward()
        torch.testing.assert_close(model.core.memory.protos, snapshot)
        torch.testing.assert_close(model.core.memory.counts, counts)
        self.assertTrue(model.core.memory.counts.sum() > 0)
        self.assertTrue(model.core.causal_module.Wq.weight.grad.abs().sum() > 0)
        model.eval()
        state = copy.deepcopy(model.state_dict())
        prediction = model(images)
        self.assertEqual(prediction.shape, (3, 3))
        self.assertTrue(torch.equal(state['core.memory.protos'], model.core.memory.protos))
        self.assertTrue(torch.equal(state['core.memory.counts'], model.core.memory.counts))
        clone = ProCI(ResToy(), 8, ['a', 'b', 'c'], SOURCE/'ProCI', SETTINGS['ProCI'])
        clone.load_state_dict(state)
        clone.eval()
        torch.testing.assert_close(prediction, clone(images))


if __name__ == '__main__':
    unittest.main()
