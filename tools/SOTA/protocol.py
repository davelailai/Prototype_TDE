"""Resolve comparison settings from the actual Q2L-TDE experiment config."""
import copy
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path

import torch
from torch import nn
from mmengine.config import Config

PROJECT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = Path(os.environ.get('SOTA_REFERENCE_ROOT', PROJECT)).resolve()


def reference_loss_class(root=REFERENCE_ROOT):
    import mmpretrain.models  # Ensure registrations happen before isolating the loss name.
    from mmpretrain.registry import MODELS
    path = Path(root).resolve()/'models/losses/MILAsyLoss.py'
    spec = importlib.util.spec_from_file_location('_sota_reference_loss', path)
    module = importlib.util.module_from_spec(spec)
    previous = MODELS.module_dict.pop('MILAsymmetricLoss', None)
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is not None:
            MODELS.module_dict['MILAsymmetricLoss'] = previous
    return module.MILAsymmetricLoss


def reference_protocol(dataset, root=REFERENCE_ROOT):
    root = Path(root).resolve()
    path = root/f'configs/{dataset}/{dataset}_CounterFactual_TDE.py'
    cfg = Config.fromfile(path)
    if cfg.model.backbone.type != 'ResNet' or cfg.model.backbone.depth != 50:
        raise ValueError('Current adapters support ResNet-50 only; do not silently substitute another backbone')
    head = cfg.model.head.task_heads.classification
    loss = copy.deepcopy(dict(head.loss))
    if loss.pop('type') != 'MILAsymmetricLoss':
        raise ValueError(f'Unsupported reference loss in {path}; audit before extending this adapter')
    cls = reference_loss_class(root)
    defaults = {name: parameter.default for name, parameter in inspect.signature(cls).parameters.items()
                if parameter.default is not inspect.Parameter.empty}
    defaults.update(loss)
    if defaults['reduction'] != 'mean':
        raise ValueError('This protocol expects the reference head mean reduction with avg_factor=batch')
    loss_path = root/'models/losses/MILAsyLoss.py'
    return dict(reference_config=str(path), reference_root=str(root),
                reference_loss_sha256=hashlib.sha256(loss_path.read_bytes()).hexdigest(),
                backbone=copy.deepcopy(dict(cfg.model.backbone)),
                loss=dict(type='MILAsymmetricLoss', **defaults),
                loss_reduction='sum over disease entries / batch size; focal weights differentiated',
                max_epochs=cfg.train_cfg.max_epochs,
                batch_size=cfg.train_dataloader.batch_size,
                optimizer=copy.deepcopy(dict(cfg.optim_wrapper.optimizer)),
                scheduler=copy.deepcopy(cfg.param_scheduler),
                normalization=copy.deepcopy(dict(cfg.data_preprocessor)),
                train_view=copy.deepcopy(cfg.view_pipeline1),
                eval_view=[copy.deepcopy(dict(op)) for op in cfg.val_dataloader.dataset.pipeline
                           if op['type'] not in ('LoadImageFromFile', 'PackInputs')],
                annotation_files={split: cfg[f'{split}_dataloader'].dataset.ann_file
                                  for split in ('train', 'val', 'test')})


class ReferenceLoss(nn.Module):
    """Exactly the loss implementation and avg_factor used by the reference head."""
    def __init__(self, protocol):
        super().__init__()
        kwargs = dict(protocol['loss'])
        kwargs.pop('type')
        self.module = reference_loss_class(protocol['reference_root'])(**kwargs)

    def forward(self, logits, labels):
        return self.module(logits, labels, avg_factor=logits.shape[0])


def configure_method(method, settings, reference, mode):
    """Keep method-specific objectives/schedules; expose exceptions explicitly."""
    settings = copy.deepcopy(settings)
    if mode == 'aligned' and method != 'CIPL':
        settings['epochs'] = reference['max_epochs']
    return settings


def audit_backbone(backbone, checkpoint, prefix='backbone'):
    """Fail on silent partial loading, rather than merely comparing checkpoint URLs."""
    if checkpoint is None:
        return dict(pretrained=False, reason='explicit scratch/debug run')
    from mmengine.runner.checkpoint import _load_checkpoint_with_prefix
    source = _load_checkpoint_with_prefix(prefix, checkpoint, map_location='cpu')
    actual = backbone.state_dict()
    required = [key for key in actual if not key.endswith('num_batches_tracked')]
    missing = [key for key in required if key not in source]
    mismatched = [key for key in required if key in source and
                  (actual[key].shape != source[key].shape or
                   not torch.equal(actual[key].detach().cpu(), source[key].to(actual[key].dtype)))]
    if missing or mismatched:
        raise RuntimeError(f'Backbone initialization mismatch: missing={missing[:8]}, different={mismatched[:8]}')
    return dict(pretrained=True, checkpoint=str(checkpoint), prefix=prefix,
                verified_tensors=len(required), missing_required=[], mismatched=[])


def alignment_report(method, mode, reference, settings, batch_size, image_size, scratch=False):
    refinements = 0
    if method == 'CIPL':
        refinements = sum(epoch >= settings['push_start'] and epoch % settings['push_interval'] == 0
                          for epoch in range(settings['epochs'])) * settings['last_epochs']
    return dict(mode=mode, reference=reference,
                classification_loss=(reference['loss'] if mode == 'aligned' and method != 'CIPL'
                                     else 'method-specific paired normal/disease CE' if method == 'CIPL'
                                     else 'previous native adaptation loss'),
                batch_matches_reference=batch_size == reference['batch_size'],
                image_size_matches_reference=image_size == reference['train_view'][1].get('scale', 448),
                epochs_matches_reference=settings['epochs'] == reference['max_epochs'],
                joint_epochs=settings['epochs'], refinement_epochs=refinements,
                total_optimizer_dataset_passes=settings['epochs']+refinements,
                pretrained=not scratch,
                exceptions=(['CIPL paired CE and staged training retained; finetuning adds dataset passes']
                            if method == 'CIPL' else []),
                optimizer_policy='method-specific; not claimed identical to Q2L-TDE',
                auxiliary_policy='method-specific mechanisms and weights retained; see README',
                compute_policy='equal epochs do not imply equal FLOPs/views/optimizer steps')
