"""Export fixed split manifests, or restore portable annotations from them."""

import argparse
import csv
import hashlib
import itertools
import json
import pickle
import re
from pathlib import Path, PurePosixPath


DATASETS = {
    'FFA': {'files': {'train': 'FFA/train_class6.pkl', 'val': 'FFA/val_class6.pkl',
                      'test': 'FFA/test_class6.pkl'},
            'patient_basis': 'original_case in FFA class6 CSVs; grouping code uses patient folders'},
    'Endo': {'files': {s: f'Endo/Endo/patient_disjoint_balanced_seed42/{s}_list.pkl'
                       for s in ('train', 'val', 'test')},
             'patient_basis': 'study_id; dataset owner confirmed one study per patient'},
    'ODIR': {'files': {'train': 'OIA-ODIR/train.pkl', 'val': 'OIA-ODIR/val.pkl',
                       'onsite_test': 'OIA-ODIR/onsite_test.pkl',
                       'offsite_test': 'OIA-ODIR/offsite_test.pkl'},
             'patient_basis': 'numeric patient ID in <ID>_left/right image filenames'},
}


def relative_image(path, project):
    source = Path(path)
    if not source.is_absolute():
        source = project / source
    return source.resolve().relative_to((project / 'data').resolve()).as_posix()


def validate_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts or '\\' in value:
        raise ValueError(f'Invalid relative image path: {value}')
    return path


def summarize(rows, classes):
    splits = sorted({r['split'] for r in rows})
    statistics = {}
    patients, images = {}, {}
    for split in splits:
        selected = [r for r in rows if r['split'] == split]
        patients[split] = {r['patient_id'] for r in selected}
        images[split] = {r['img_path'] for r in selected}
        if len(images[split]) != len(selected):
            raise ValueError(f'Duplicate image in {split}')
        if '' in patients[split]:
            raise ValueError(f'Missing patient ID in {split}')
        statistics[split] = {
            'patients': len(patients[split]), 'images': len(selected),
            'positive_images': {c: sum(int(r[c]) for r in selected) for c in classes},
        }
    overlaps = {}
    for a, b in itertools.combinations(splits, 2):
        overlaps[f'{a}-{b}'] = {'patients': len(patients[a] & patients[b]),
                                'images': len(images[a] & images[b])}
    if any(n for v in overlaps.values() for n in v.values()):
        raise ValueError(f'Cross-split overlap: {overlaps}')
    return statistics, overlaps


def export(args):
    project = args.project_root.resolve()
    if any((args.manifest_root / d).exists() for d in args.datasets):
        raise ValueError('Manifest destination already exists; use a fresh directory.')
    prepared = []
    for dataset in args.datasets:
        config = DATASETS[dataset]
        rows, hashes, classes = [], {}, None
        for split, filename in config['files'].items():
            source = project / 'data' / filename
            payload = source.read_bytes()
            annotations = pickle.loads(payload)
            current_classes = list(annotations['metainfo']['classes'])
            if classes is not None and classes != current_classes:
                raise ValueError(f'{dataset}: inconsistent class order')
            classes = current_classes
            hashes[filename] = hashlib.sha256(payload).hexdigest()
            ffa_patients = {}
            if dataset == 'FFA':
                with source.with_suffix('.csv').open(encoding='gbk', newline='') as f:
                    for r in csv.DictReader(f):
                        key = relative_image(r['image_path'], project)
                        if key in ffa_patients:
                            raise ValueError(f'Duplicate FFA CSV image: {key}')
                        ffa_patients[key] = r['original_case']
            for item in annotations['data_list']:
                image = relative_image(item['img_path'], project)
                validate_path(image)
                if dataset == 'FFA':
                    patient = ffa_patients[image]
                elif dataset == 'ODIR':
                    match = re.fullmatch(r'(\d+)_(left|right)\.[^.]+', PurePosixPath(image).name)
                    if not match:
                        raise ValueError(f'Cannot identify ODIR patient: {image}')
                    patient = match.group(1)
                else:
                    patient = item['group_id']
                labels = list(item['gt_score'])
                if len(labels) != len(classes) or any(v not in (0, 1) for v in labels):
                    raise ValueError(f'Invalid labels: {image}')
                rows.append({'split': split, 'patient_id': str(patient), 'img_path': image,
                             **dict(zip(classes, map(int, labels)))})
        stats, overlaps = summarize(rows, classes)
        report = {'dataset': dataset, 'classes': classes, 'annotation_files': config['files'],
                  'image_path_base': 'external image root (the directory containing FFA, Endo, OIA-ODIR)',
                  'patient_id_basis': config['patient_basis'],
                  'assignment': 'new patient-disjoint balanced split' if dataset == 'Endo' else 'existing assignments preserved',
                  'official_split_membership_independently_verified': False,
                  'source_annotation_sha256': hashes, 'split_statistics': stats, 'overlap': overlaps}
        if dataset == 'Endo':
            origin = json.loads((project / 'data/Endo/Endo/patient_disjoint_balanced_seed42/split_report.json').read_text())
            report['assignment_details'] = {k: origin[k] for k in (
                'seed', 'ratios', 'assignment_strategy', 'candidate_count', 'balance_objective',
                'coverage_constraints', 'verification_basis')}
        prepared.append((dataset, rows, report))
    for dataset, rows, report in prepared:
        folder = args.manifest_root / dataset
        folder.mkdir(parents=True)
        manifest = folder / 'split_manifest.csv'
        with manifest.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['split', 'patient_id', 'img_path'] + report['classes'])
            writer.writeheader()
            writer.writerows(rows)
        report['manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        (folder / 'split_report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(dataset, json.dumps(report['split_statistics']))


def materialize(args):
    prepared = []
    for dataset in args.datasets:
        folder = args.manifest_root / dataset
        report = json.loads((folder / 'split_report.json').read_text())
        manifest = folder / 'split_manifest.csv'
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != report['manifest_sha256']:
            raise ValueError(f'Manifest checksum mismatch: {dataset}')
        with manifest.open(newline='') as f:
            rows = list(csv.DictReader(f))
        classes = report['classes']
        if set(r['split'] for r in rows) != set(report['annotation_files']):
            raise ValueError(f'Unexpected split names: {dataset}')
        for r in rows:
            validate_path(r['img_path'])
            if any(r[c] not in ('0', '1') for c in classes):
                raise ValueError(f'Invalid labels: {r["img_path"]}')
        statistics, overlaps = summarize(rows, classes)
        if statistics != report['split_statistics'] or overlaps != report['overlap']:
            raise ValueError(f'Manifest statistics mismatch: {dataset}')
        for split, filename in report['annotation_files'].items():
            validate_path(filename)
            destination = args.out_root / filename
            if destination.exists():
                raise FileExistsError(f'{destination} already exists; use a fresh --out-root.')
            records = []
            for r in rows:
                if r['split'] != split:
                    continue
                image = args.image_root.resolve() / r['img_path']
                if not args.skip_image_check and not image.is_file():
                    raise FileNotFoundError(image)
                item = {'img_path': r['img_path'] if args.relative_images else str(image),
                        'gt_score': [float(r[c]) for c in classes], 'patient_id': r['patient_id']}
                if dataset == 'Endo':
                    item.update(group_id=r['patient_id'], study_id=r['patient_id'])
                records.append(item)
            prepared.append((destination, {'metainfo': {'classes': classes}, 'data_list': records}))
    for destination, annotations in prepared:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('wb') as f:
            pickle.dump(annotations, f)
        print(f'{destination}: {len(annotations["data_list"])} images')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    exporter = subparsers.add_parser('export', help='Export the existing local annotations.')
    exporter.add_argument('--project-root', type=Path, default=Path('.'))
    importer = subparsers.add_parser('materialize', help='Generate local PKLs from committed CSV manifests.')
    importer.add_argument('--image-root', type=Path, required=True)
    importer.add_argument('--out-root', type=Path, default=Path('data'))
    importer.add_argument('--relative-images', action='store_true',
                          help='Store relative paths; pass dataset.data_prefix.img_path at training/test time.')
    importer.add_argument('--skip-image-check', action='store_true', help='For metadata-only validation.')
    for p in (exporter, importer):
        p.add_argument('--manifest-root', type=Path, default=Path('splits'))
        p.add_argument('--datasets', nargs='+', choices=list(DATASETS), default=list(DATASETS))
    args = parser.parse_args()
    (export if args.command == 'export' else materialize)(args)


if __name__ == '__main__':
    main()
