"""Synthetic GPU memory check; never trains/evaluates on medical data."""
import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn

from adapters import make_model
from protocol import ReferenceLoss, reference_optimization, reference_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--size', type=int, default=448)
    parser.add_argument('--reference-root', required=True)
    parser.add_argument('--without-checkpointing', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable')
    torch.manual_seed(0)
    reference = reference_protocol('FFA', args.reference_root)
    settings = dict(activation_checkpointing=not args.without_checkpointing)
    model = make_model('MLC-PAT', list(range(6)), '.', settings, args.size, None,
                       backbone_config=reference['backbone'], classification_loss=ReferenceLoss(reference)).cuda().train()
    ema = copy.deepcopy(model).requires_grad_(False)
    optimizer, _ = reference_optimization(model.parameters(), reference)
    views = [torch.randn(args.batch_size, 3, args.size, args.size, device='cuda') for _ in range(5)]
    labels = torch.randint(0, 2, (args.batch_size, 6), device='cuda').float()
    torch.cuda.reset_peak_memory_stats()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(views, labels, step)
        expected = [(m, {n: b.detach().clone() for n, b in m.named_buffers(recurse=False)})
                    for m in model.backbone.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        model.backward_loss(loss)
        for module, buffers in expected:
            for name, value in buffers.items():
                torch.testing.assert_close(getattr(module, name), value, rtol=0, atol=0)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        print(f'Step {step}: loss={float(loss):.6f}', flush=True)
    torch.cuda.synchronize()
    result = dict(gpu=torch.cuda.get_device_name(), batch_size=args.batch_size, views=5, size=args.size,
                  checkpointing=settings['activation_checkpointing'], dtype='float32',
                  optimizer='reference SGD', resident_ema_parameters=sum(p.numel() for p in ema.parameters()),
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                  batchnorm_recompute_parity=True, finite_gradients=True, synthetic_data=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
