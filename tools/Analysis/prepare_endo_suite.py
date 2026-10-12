"""Prepare a fixed-split Endo experiment array without starting training."""

import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path

from mmengine.config import Config


MAIN_CONFIGS = {
    'Base': 'Endo_baseline.py', 'Asy': 'Endo_AsyLoss.py', 'dyGCN': 'Endo_GCNhead.py',
    'MLDecoder': 'Endo_MLDecoder.py', 'Q2L': 'Endo_Q2L.py',
}
COMPONENTS = {
    'baseline': (0, 0, 0), 'only_lpos': (1, 0, 0), 'only_lneg': (0, 1, 0),
    'only_ortho': (0, 0, .1), 'no_ortho': (1, 1, 0), 'no_lneg': (1, 0, .1),
    'no_lpos': (0, 1, .1), 'all_loss': (1, 1, .1),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, default=Path('.'))
    p.add_argument('--baseline-root', type=Path, required=True,
                   help='Existing full implementation containing the five main baseline configs.')
    p.add_argument('--image-root', type=Path, required=True)
    p.add_argument('--annotation-root', type=Path, default=Path('data'))
    p.add_argument('--suite-dir', type=Path, default=Path('work_dirs/Endo_patient_disjoint_v1'))
    p.add_argument('--groups', nargs='+', choices=['main', 'component', 'margin', 'lambda'],
                   default=['main', 'component', 'margin', 'lambda'])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    args = p.parse_args()
    root = args.project_root.resolve()
    legacy = args.baseline_root.resolve()
    suite = args.suite_dir.resolve()
    if suite.exists():
        raise FileExistsError(f'{suite} exists. Reuse its plan for submission or choose a fresh suite directory.')
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Duplicate seeds.')
    annotation_root = args.annotation_root.resolve()
    report = json.loads((root / 'splits/Endo/split_report.json').read_text())
    manifest = root / 'splits/Endo/split_manifest.csv'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != report['manifest_sha256']:
        raise ValueError('Endo manifest checksum mismatch.')
    groups, annotation_hashes = {}, {}
    from csv import DictReader
    with manifest.open(newline='') as f:
        assignments = list(DictReader(f))
    for split, filename in report['annotation_files'].items():
        path = annotation_root / filename
        payload = path.read_bytes()
        records = pickle.loads(payload)['data_list']
        expected = [r for r in assignments if r['split'] == split]
        if len(records) != len(expected):
            raise ValueError(f'{split} annotation count mismatch.')
        groups[split] = set()
        for item, row in zip(records, expected):
            image = Path(item['img_path'])
            if image.is_absolute():
                image = image.resolve()
                expected_image = args.image_root.resolve() / row['img_path']
            else:
                image = args.image_root.resolve() / image
                expected_image = args.image_root.resolve() / row['img_path']
            if image != expected_image or list(item['gt_score']) != [float(row[c]) for c in report['classes']]:
                raise ValueError(f'{split}: annotation differs from committed manifest.')
            if not image.is_file():
                raise FileNotFoundError(image)
            groups[split].add(row['patient_id'])
        annotation_hashes[str(path)] = hashlib.sha256(payload).hexdigest()
    for a in groups:
        for b in groups:
            if a != b and groups[a] & groups[b]:
                raise ValueError('Patient overlap.')
    template = Config.fromfile(str(root / 'configs/Endo/Endo_CounterFactual_TDE.py'))
    template.model.head.task_heads.classification.update(
        prototype_init='query_mlp', prototype_contextual=False,
        intervention_mode='proj_gate', train_with_TDE=False)
    shared = {key: copy.deepcopy(template[key]) for key in (
        'data_preprocessor', 'train_dataloader', 'val_dataloader', 'test_dataloader',
        'optim_wrapper', 'param_scheduler', 'env_cfg', 'default_hooks', 'visualizer', 'default_scope')}
    for split, filename in report['annotation_files'].items():
        dataset = shared[f'{split}_dataloader']['dataset']
        dataset['ann_file'] = str(annotation_root / filename)
        dataset['data_prefix'] = {'img_path': str(args.image_root.resolve())}
    shared['default_hooks']['checkpoint'].update(
        save_best='multi-class/mAUC_multiclass', rule='greater', max_keep_ckpts=1)
    # Select all methods by validation AUC from their original predictions.
    validation = [dict(type='RankingROC_AUC', Validation_mode='original')]
    evaluation = validation + [dict(type='MultiLabel_AveragePrecision', Validation_mode='original'),
        dict(type='RankingMultiLabelMetric', items=['precision', 'recall', 'f1-score', 'support'],
             thr=.5, Validation_mode='original')]
    variants = {}

    def add_tde(pos, neg, ortho, margin, alias):
        key = ('tde', float(pos), float(neg), float(ortho), float(margin))
        if key not in variants:
            cfg = copy.deepcopy(template)
            cfg.model.head.task_heads.classification.update(
                lambda_interv_pos=float(pos), lambda_interv_neg=float(neg),
                orthogonality_weight=float(ortho), intervention_margin=float(margin))
            variants[key] = {'config': cfg, 'code_root': root, 'aliases': []}
        variants[key]['aliases'].append(alias)

    if 'component' in args.groups:
        for name, weights in COMPONENTS.items():
            add_tde(*weights, .5, f'Component_Ablation/{name}')
    if 'main' in args.groups:
        add_tde(1, 1, .1, .5, 'Main_Comparison/Q2L_TDE')
        for method, filename in MAIN_CONFIGS.items():
            source = legacy / 'configs/Endo' / filename
            cfg = Config.fromfile(str(source))
            cfg.model.backbone = copy.deepcopy(template.model.backbone)
            cfg.model.head.task_heads.classification.update(num_classes=4, in_channels=2048)
            cfg.data_preprocessor = copy.deepcopy(template.data_preprocessor)
            variants[('main', method)] = {'config': cfg, 'code_root': legacy,
                                         'aliases': [f'Main_Comparison/{method}']}
    if 'margin' in args.groups:
        for margin in (.1, .3, .5, .7, .9, 1.1):
            add_tde(1, 1, .1, margin, f'Margin_Ablation/margin{margin:.1f}')
    if 'lambda' in args.groups:
        for pos in (.2, .4, .6, .8, 1.0):
            for neg in (.2, .4, .6, .8, 1.0):
                add_tde(pos, neg, .1, .5, f'Lambda_Ablation/lpos{pos:.1f}_lneg{neg:.1f}')
    tasks, prepared = [], []
    for variant in variants.values():
        code_root = variant['code_root']
        for tool in ('Train.py', 'Test.py'):
            if not (code_root / 'tools' / tool).is_file():
                raise FileNotFoundError(code_root / 'tools' / tool)
        for seed in args.seeds:
            cfg = copy.deepcopy(variant['config'])
            for key, value in shared.items():
                cfg[key] = copy.deepcopy(value)
            cfg.val_evaluator = copy.deepcopy(validation)
            cfg.test_evaluator = copy.deepcopy(evaluation)
            cfg.train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
            cfg.val_cfg, cfg.test_cfg = {}, {}
            cfg.randomness = dict(seed=seed, deterministic=False)
            cfg.resume, cfg.load_from = False, None
            work = suite / variant['aliases'][0] / f'fold{seed}'
            cfg.work_dir = str(work)
            config = work / 'run.py'
            tasks.append({'index': len(tasks), 'seed': seed, 'config': str(config),
                          'code_root': str(code_root), 'work_dir': str(work),
                          'aliases': variant['aliases']})
            prepared.append((config, cfg))
    for path, cfg in prepared:
        path.parent.mkdir(parents=True)
        cfg.dump(str(path))
    plan = {'tasks': tasks, 'seeds': args.seeds, 'groups': args.groups,
            'manifest_sha256': report['manifest_sha256'], 'annotation_sha256': annotation_hashes,
            'image_root': str(args.image_root.resolve()),
            'note': 'Five seeds on one fixed patient split, not five-fold cross-validation. '
                    'Component baseline is the zero-loss query geometry control; '
                    'Main_Comparison/Q2L uses the original Q2L implementation.'}
    (suite / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(f'Prepared {len(variants)} unique settings, {len(tasks)} training+test tasks.')
    print(f'Plan: {suite / "plan.json"}')
    print('Repeated full-model settings are shared through task aliases.')


if __name__ == '__main__':
    main()
