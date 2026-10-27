"""Exact scheduler parity and checkpoint recomputation BatchNorm safeguards."""
import copy
import unittest

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from mmengine.registry import PARAM_SCHEDULERS
import mmengine.optim  # Register scheduler implementations.

from adapters import PAT, make_model
from protocol import reference_optimization, reference_protocol


class CheckpointBackbone(nn.Module):
    def __init__(self, enabled=False):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.bn = nn.BatchNorm2d(4)
        self.enabled = enabled

    def forward(self, x):
        x = self.conv(x)
        return checkpoint(self.bn, x, use_reentrant=True) if self.enabled else self.bn(x)


class SharedOptimizationTest(unittest.TestCase):
    def test_real_resnet_checkpoint_flags(self):
        ref = reference_protocol('FFA')
        model = make_model('MLC-PAT', ['a', 'b'], '.', dict(activation_checkpointing=True),
                           32, None, backbone_config=ref['backbone'])
        self.assertTrue(model.memory_checkpointing)
        blocks = [m for m in model.backbone.modules() if m.__class__.__name__ == 'Bottleneck']
        self.assertTrue(blocks)
        self.assertTrue(all(m.with_cp for m in blocks))
        model.train()
        loss = model.loss([torch.randn(2, 3, 32, 32) for _ in range(5)],
                          torch.tensor([[1., 0.], [0., 1.]]), 0)
        expected = {name: value.detach().clone() for name, value in model.backbone.named_buffers()}
        model.backward_loss(loss)
        for name, value in model.backbone.named_buffers():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        self.assertTrue(model.backbone.net.conv1.weight.grad is not None)

    def test_reference_scheduler_trace(self):
        for dataset in ['FFA', 'Endo', 'ODIR']:
            ref = reference_protocol(dataset)
            parameter = nn.Parameter(torch.ones(1))
            optimizer, schedulers = reference_optimization([parameter], ref)
            native = copy.deepcopy(ref['optimizer'])
            native.pop('type')
            direct = torch.optim.SGD([nn.Parameter(torch.ones(1))], **native)
            expected = [PARAM_SCHEDULERS.build(dict(entry, optimizer=direct)) for entry in ref['scheduler']]
            for epoch in range(ref['max_epochs']):
                self.assertEqual(optimizer.param_groups[0]['lr'], direct.param_groups[0]['lr'])
                optimizer.step()
                direct.step()
                for scheduler in schedulers:
                    scheduler.step()
                for scheduler in expected:
                    scheduler.step()
            for key in ['momentum', 'weight_decay', 'nesterov', 'dampening']:
                self.assertEqual(optimizer.defaults[key], direct.defaults[key])

    def test_checkpoint_gradient_and_batchnorm_parity(self):
        torch.manual_seed(3)
        ordinary = PAT(CheckpointBackbone(), 4, 3)
        saved = copy.deepcopy(ordinary)
        saved.backbone.enabled = True
        saved.memory_checkpointing = True
        views = [torch.randn(2, 3, 8, 8) for _ in range(5)]
        labels = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
        a, b = ordinary.loss(views, labels, 0), saved.loss(views, labels, 0)
        torch.testing.assert_close(a, b)
        ordinary.backward_loss(a)
        saved.backward_loss(b)
        for (_, p), (_, q) in zip(ordinary.named_parameters(), saved.named_parameters()):
            torch.testing.assert_close(p.grad, q.grad)
        for (_, p), (_, q) in zip(ordinary.named_buffers(), saved.named_buffers()):
            torch.testing.assert_close(p, q)
        self.assertEqual(saved.backbone.bn.num_batches_tracked.item(), 1)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
