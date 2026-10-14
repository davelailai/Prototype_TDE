"""Numerical parity with the actual project loss, not just its configuration."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import asymmetric_loss
from protocol import (ReferenceLoss, alignment_report, audit_backbone, configure_method,
                      reference_loss_class, reference_protocol)
from data import ReferenceView
from PIL import Image

torch.set_num_threads(2)


class ProtocolTest(unittest.TestCase):
    def test_reference_configs(self):
        for dataset in ('FFA', 'Endo', 'ODIR'):
            p = reference_protocol(dataset)
            self.assertEqual(p['backbone']['depth'], 50)
            self.assertEqual(p['loss']['gamma_pos'], 0.)
            self.assertEqual(p['loss']['gamma_neg'], 4.)
            self.assertEqual(p['batch_size'], 32)
            self.assertEqual(p['max_epochs'], 100)

    def test_loss_and_gradient_equal_project(self):
        p = reference_protocol('FFA')
        kwargs = dict(p['loss'])
        kwargs.pop('type')
        direct = reference_loss_class()(**kwargs)
        aligned = ReferenceLoss(p)
        for batch, classes in ((2, 6), (4, 7)):
            logits = torch.randn(batch, classes, requires_grad=True)
            target = torch.randint(0, 2, logits.shape).float()
            expected = direct(logits, target, avg_factor=batch)
            actual = aligned(logits, target)
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            gradients = [torch.autograd.grad(value, logits, retain_graph=True)[0] for value in (expected, actual)]
            torch.testing.assert_close(*gradients, rtol=0, atol=0)

    def test_batch_not_class_mean(self):
        aligned = ReferenceLoss(reference_protocol('FFA'))
        x = torch.tensor([[.2, -.5, 1.]], requires_grad=True)
        y = torch.tensor([[1., 0., 1.]])
        torch.testing.assert_close(aligned(x, y), aligned(x.repeat(5, 1), y.repeat(5, 1)))
        # Native detached focal and aligned differentiable focal have equal
        # values after batch scaling but DIFFERENT gradients.
        z = x.repeat(2, 1).detach().requires_grad_(True)
        t = y.repeat(2, 1)
        a, b = aligned(z, t), asymmetric_loss(z, t)/2
        torch.testing.assert_close(a, b)
        ga = torch.autograd.grad(a, z, retain_graph=True)[0]
        gb = torch.autograd.grad(b, z)[0]
        self.assertFalse(torch.allclose(ga, gb))

    def test_backbone_audit_success_and_failure(self):
        model = nn.Linear(3, 2)
        state = {'backbone.'+k: v.clone() for k, v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'weights.pth'
            torch.save(dict(state_dict=state), path)
            report = audit_backbone(model, str(path))
            self.assertEqual(report['verified_tensors'], 2)
            with torch.no_grad():
                model.weight.add_(1)
            with self.assertRaisesRegex(RuntimeError, 'initialization mismatch'):
                audit_backbone(model, str(path))
            state.pop('backbone.bias')
            torch.save(dict(state_dict=state), path)
            with self.assertRaisesRegex(RuntimeError, 'missing='):
                audit_backbone(model, str(path))

    def test_eval_transform_matches_mm_ops(self):
        reference = reference_protocol('FFA')
        view = ReferenceView(reference['eval_view'], 64)
        image = Image.fromarray(np.random.default_rng(2).integers(0, 256, (120, 80, 3), dtype=np.uint8))
        expected = np.asarray(view(image))
        array = np.asarray(image)[:, :, ::-1].copy()
        data = dict(img=array, img_shape=array.shape[:2], ori_shape=array.shape[:2])
        for op in view.operations:
            data = op(data)
        np.testing.assert_array_equal(expected, data['img'][:, :, ::-1])
        self.assertEqual(expected.shape[:2], (64, 64))

    def test_native_and_cipl_exceptions(self):
        reference = reference_protocol('FFA')
        self.assertEqual(configure_method('ProCI', dict(epochs=80), reference, 'aligned')['epochs'], 100)
        self.assertEqual(configure_method('ProCI', dict(epochs=80), reference, 'native')['epochs'], 80)
        settings = dict(epochs=41, push_start=35, push_interval=5, last_epochs=6)
        report = alignment_report('CIPL', 'aligned', reference, settings, 8, 448)
        self.assertEqual(report['total_optimizer_dataset_passes'], 53)
        self.assertFalse(report['batch_matches_reference'])
        self.assertFalse(report['epochs_matches_reference'])
        self.assertTrue(report['exceptions'])


if __name__ == '__main__':
    unittest.main()
