"""Execute a generated BankCausal config with its official models and hooks."""
import argparse
import importlib
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.hooks import Hook
from mmengine.registry import HOOKS, METRICS

from common import metrics
from fetch_sources import verify
from protocol import audit_backbone, reference_loss_class


@HOOKS.register_module()
class SOTABankStateHook(Hook):
    def before_train(self, runner):
        init = runner.cfg.model.backbone.get('init_cfg')
        result = audit_backbone(runner.model.backbone, init['checkpoint'] if init else None)
        (Path(runner.work_dir)/'initialization_audit.json').write_text(json.dumps(result, indent=2))

    def before_save_checkpoint(self, runner, checkpoint):
        head = runner.model.head
        checkpoint['sota_bank_state'] = dict(current_epoch=head.current_epoch,
            ema_prototypes=head.ema_prototypes.detach().cpu().clone())

    def after_load_checkpoint(self, runner, checkpoint):
        state = checkpoint.get('sota_bank_state')
        if state is None:
            raise ValueError('Checkpoint lacks BankCausal inference state; use this entrypoint to train')
        head = runner.model.head
        head.current_epoch = state['current_epoch']
        head.warmup_epoch = runner.model.warmup_epoch
        head.ema_prototypes = state['ema_prototypes'].to(next(head.parameters()).device)


@METRICS.register_module()
class SOTAClassificationMetric(BaseMetric):
    default_prefix = 'SOTA'

    def __init__(self, output=None, **kwargs):
        super().__init__(**kwargs)
        self.output = output

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            sample = sample.to_dict() if hasattr(sample, 'to_dict') else sample
            self.results.append(dict(pred_score=sample['pred_score'].detach().cpu().numpy(),
                                     gt_score=sample['gt_score'].detach().cpu().numpy(),
                                     img_path=sample.get('img_path'), group_id=sample.get('group_id')))

    def compute_metrics(self, results):
        output = metrics(np.stack([r['gt_score'] for r in results]), np.stack([r['pred_score'] for r in results]))
        if self.output:
            root = Path(self.output)
            root.mkdir(parents=True, exist_ok=True)
            with (root/'pred_results.pkl').open('wb') as handle:
                pickle.dump(results, handle)
            (root/'test_metrics.json').write_text(json.dumps(output, indent=2))
        return {key: output[key] for key in ('mAUC', 'mAP', 'MAP', 'MAR', 'MAF1')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--source-root', default='third_party/sota')
    parser.add_argument('--check', action='store_true', help='Build model/runner, but do not train')
    args = parser.parse_args()
    source = verify(args.source_root, 'BankCausal')
    config = Path(args.config).resolve()
    sys.path.insert(0, str(source))
    os.chdir(source)
    from mmpretrain.utils import register_all_modules
    register_all_modules(init_default_scope=False)
    from mmpretrain.registry import MODELS
    if MODELS.get('MultiLabelClsDataPreprocessor') is None:
        # The public BankCausal repository omits this multi-view preprocessor;
        # use the implementation already maintained by the current project.
        path = Path(__file__).resolve().parents[2]/'models/utils/data_preprossor.py'
        spec = importlib.util.spec_from_file_location('_sota_data_preprocessor', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    for name in ('models', 'datasets', 'evaluation'):
        importlib.import_module(name)
    if not torch.cuda.is_available():
        from models.losses.swav_loss import MultiLabelSwAVLoss
        initialize = MultiLabelSwAVLoss.orthogonal_l2_normalized_init
        def cpu_initialize(self, num_prototypes, feat_dim, device='cpu'):
            return initialize(self, num_prototypes, feat_dim, device=device)
        MultiLabelSwAVLoss.orthogonal_l2_normalized_init = cpu_initialize
    from mmengine.config import Config
    from mmengine.runner import Runner
    cfg = Config.fromfile(config)
    if cfg.get('sota_alignment', {}).get('mode') == 'aligned':
        root = cfg.sota_alignment.reference.reference_root
        MODELS.register_module(name='SOTAAlignedLoss', module=reference_loss_class(root))
    cfg['val_evaluator'] = dict(type='SOTAClassificationMetric')
    cfg['test_evaluator'] = dict(type='SOTAClassificationMetric', output=cfg.work_dir)
    # Upstream's custom test loop forces epoch=warmup+100. Use the SAME branch
    # that was selected on validation, and persist its non-buffer EMA state.
    cfg['test_cfg'] = dict(type='TestLoop')
    cfg.setdefault('custom_hooks', []).append(dict(type='SOTABankStateHook', priority='LOW'))
    cfg.default_hooks.checkpoint.save_best = 'SOTA/mAUC'
    runner = Runner.from_cfg(cfg)
    if args.check:
        print('BankCausal model and runner built; training not started')
        return
    runner.train()
    best = next((hook.best_ckpt_path for hook in runner.hooks if getattr(hook, 'best_ckpt_path', None)), None)
    if best is None:
        raise RuntimeError('No validation-selected checkpoint produced')
    runner.load_checkpoint(best)
    runner.test()


if __name__ == '__main__':
    main()
