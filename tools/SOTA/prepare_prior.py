"""Generate a portable config for the official BankCausal implementation."""
import argparse
import copy
import json
import pickle
import shlex
import sys
from pathlib import Path

from mmengine.config import Config

from data import ANNOTATIONS, read_annotations, resolve_image
from fetch_sources import verify
from protocol import REFERENCE_ROOT, alignment_report, reference_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['BankCausal'], default='BankCausal')
    parser.add_argument('--dataset', choices=list(ANNOTATIONS), required=True)
    parser.add_argument('--source-root', default='third_party/sota')
    parser.add_argument('--annotation-root', default='data')
    parser.add_argument('--image-root', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--checkpoint')
    parser.add_argument('--protocol', choices=['aligned', 'native'], default='aligned')
    parser.add_argument('--reference-root', default=str(REFERENCE_ROOT))
    parser.add_argument('--train-ann')
    parser.add_argument('--val-ann')
    parser.add_argument('--test-ann')
    args = parser.parse_args()
    reference = reference_protocol(args.dataset, args.reference_root)
    args.batch_size = args.batch_size or reference['batch_size']
    args.checkpoint = args.checkpoint or reference['backbone']['init_cfg']['checkpoint']
    source = verify(args.source_root, args.method)
    # The FFA recipe is adapted to each disease vocabulary, rather than silently
    # inheriting the old Endo split or an architecture different from Q2L-TDE.
    base = source/'configs/FFA/FFA_BankCausal.py'
    cfg = Config.fromfile(base)
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    if (work/'resolved.py').exists():
        raise FileExistsError(f'Refusing to overwrite {work}/resolved.py')
    classes, seen, seen_groups = None, set(), set()
    for split, relative in zip(('train', 'val', 'test'), ANNOTATIONS[args.dataset]):
        ann = Path(getattr(args, split+'_ann') or Path(args.annotation_root)/relative)
        names, rows = read_annotations(ann)
        if classes is not None and names != classes:
            raise ValueError('Class orders differ between annotation files')
        classes = names
        converted = copy.deepcopy(rows)
        for row in converted:
            row['img_path'] = str(resolve_image(row['img_path'], args.image_root))
        paths = {r['img_path'] for r in converted}
        if seen & paths:
            raise ValueError('Image overlap between annotation splits')
        seen.update(paths)
        groups = {r['group_id'] for r in converted if r.get('group_id') is not None}
        if seen_groups & groups:
            raise ValueError('Patient/study overlap between annotation splits')
        seen_groups.update(groups)
        destination = work/f'{split}_annotations.pkl'
        with destination.open('wb') as handle:
            pickle.dump(dict(metainfo=dict(classes=classes), data_list=converted), handle)
        dl = cfg[f'{split}_dataloader']
        dl['batch_size'], dl['num_workers'] = args.batch_size, args.workers
        dl['persistent_workers'] = args.workers > 0
        dl['dataset']['ann_file'] = str(destination)
        dl['dataset']['data_prefix'] = ''
        dl['dataset'].pop('data_root', None)
        dl['dataset']['metainfo'] = dict(classes=classes)
        if split != 'train':
            dl['dataset']['pipeline'] = [dict(type='LoadImageFromFile'), dict(type='Resize', scale=512),
                                         dict(type='CenterCrop', crop_size=448), dict(type='PackInputs')]
        elif args.protocol == 'aligned':
            dl['dataset']['pipeline'] = [dict(type='LoadImageFromFile'),
                dict(type='MultiLabelMultiView', num_views=2, transforms=[copy.deepcopy(reference['train_view'])]),
                dict(type='PackInputs')]
    cfg.model['backbone'] = dict(type='ResNet', depth=50, num_stages=4, out_indices=(3,), style='pytorch',
                                  init_cfg=dict(type='Pretrained', checkpoint=args.checkpoint, prefix='backbone'))
    cfg.model.head['num_classes'] = len(classes)
    cfg.model.head['in_channels'] = 2048
    cfg.model.head['img_size'] = 448
    if args.protocol == 'aligned':
        cfg.model['backbone'] = copy.deepcopy(reference['backbone'])
        cfg.model.backbone.init_cfg.checkpoint = args.checkpoint
        cfg['data_preprocessor'] = copy.deepcopy(reference['normalization'])
        cfg.model['data_preprocessor'] = dict(reference['normalization'], type='MultiLabelClsDataPreprocessor')
        cfg.model.head['loss'] = dict(reference['loss'], type='SOTAAlignedLoss')
        cfg['train_cfg']['max_epochs'] = reference['max_epochs']
    cfg.model.head.swav_loss['num_prototypes'] = 2*len(classes)
    cfg['custom_hooks'] = [hook for hook in cfg.get('custom_hooks', [])
                           if hook.get('type') != 'PrototypeClusteringHook']
    for hook in cfg.custom_hooks:
        if hook['type'] == 'MultiLabelSwAVHook':
            hook['batch_size'] = args.batch_size
    cfg.val_evaluator = [dict(type='RankingROC_AUC', Validation_mode='original')]
    cfg.test_evaluator = [dict(type='RankingROC_AUC', Validation_mode='original')]
    cfg['randomness'] = dict(seed=args.seed, deterministic=False)
    cfg['work_dir'] = str(work)
    cfg['default_hooks']['checkpoint']['save_best'] = 'multi-class/mAUC_multiclass'
    cfg['default_hooks']['checkpoint']['rule'] = 'greater'
    cfg['visualizer'] = dict(type='UniversalVisualizer', vis_backends=[dict(type='LocalVisBackend')])
    cfg['sota_alignment'] = alignment_report('BankCausal', args.protocol, reference,
                                            dict(epochs=cfg.train_cfg.max_epochs), args.batch_size, 448)
    cfg.dump(work/'resolved.py')
    script = Path(__file__).with_name('train_bankcausal.py').resolve()
    command = [sys.executable, str(script), str(work/'resolved.py'), '--source-root', str(Path(args.source_root).resolve())]
    (work/'launch.json').write_text(json.dumps(dict(cwd=str(Path.cwd()), command=command,
                                                 method=args.method, dataset=args.dataset,
                                                 protocol='controlled ResNet-50/SwAV adaptation'), indent=2))
    print(shlex.join(command))


if __name__ == '__main__':
    main()
