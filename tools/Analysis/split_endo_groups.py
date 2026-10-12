"""Create disjoint Endo splits, with explicit study/patient provenance."""

import argparse
import csv
import hashlib
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path


CLASSES = ['ulcer', 'erosion', 'polyp', 'tumor']


def partition_groups(groups, seed, strategy, candidates):
    ids = sorted(groups)
    n_train, n_val = int(len(ids) * .7), int(len(ids) * .1)
    rng = random.Random(seed)
    # Balance image counts, per-label positive images, and positive patients.
    features = {}
    for group, rows in groups.items():
        positives = [sum(int(r[c]) for r in rows) for c in CLASSES]
        features[group] = [len(rows)] + positives + [int(n > 0) for n in positives]
    totals = [sum(f[k] for f in features.values()) for k in range(9)]
    best, best_score = None, float('inf')
    for _ in range(candidates if strategy == 'balanced' else 1):
        shuffled = ids.copy()
        rng.shuffle(shuffled)
        partitions = {'train': shuffled[:n_train], 'val': shuffled[n_train:n_train+n_val],
                      'test': shuffled[n_train+n_val:]}
        score = 0.0
        eligible = True
        for split, fraction in zip(partitions.values(), (.7, .1, .2)):
            counts = [sum(features[g][k] for g in split) for k in range(9)]
            if strategy == 'balanced' and (
                any(n < 10 or counts[0] - n < 10 for n in counts[1:5])
                or any(n < 2 for n in counts[5:])
            ):
                eligible = False
                break
            score += sum(((count - fraction * total) / max(fraction * total, 1)) ** 2
                         for count, total in zip(counts, totals))
        if eligible and score < best_score:
            best, best_score = partitions, score
    if best is None:
        raise ValueError('No split satisfies the fixed label coverage requirements. '
                         'Reconsider the split proportions or use grouped cross-validation.')
    return best, best_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=Path('data/Endo/Endo/Endo_published_4d.csv'))
    parser.add_argument('--images-dir', type=Path, default=Path('data/Endo/Endo/images'))
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--strategy', choices=['random', 'balanced'], default='balanced')
    parser.add_argument('--candidates', type=int, default=5000,
                        help='Fixed metadata-only candidate count for balanced assignment.')
    parser.add_argument('--patient-map', type=Path, help='CSV with study_id and patient_id columns.')
    parser.add_argument('--confirm-study-is-patient', action='store_true',
                        help='Use only with independently confirmed one-study-per-patient metadata.')
    args = parser.parse_args()
    if args.candidates < 1:
        raise ValueError('--candidates must be positive.')
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError('Output directory is not empty; existing splits will not be overwritten.')
    with args.csv.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if len({r['img_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate img_id in source metadata.')
    patient_map = {}
    if args.patient_map:
        with args.patient_map.open(newline='') as f:
            for r in csv.DictReader(f):
                study, patient = r['study_id'].strip(), r['patient_id'].strip()
                if not patient or (study in patient_map and patient_map[study] != patient):
                    raise ValueError(f'Invalid or conflicting patient mapping for {study}')
                patient_map[study] = patient
    groups = defaultdict(list)
    for r in rows:
        if not r['study_id'].strip():
            raise ValueError('Empty study_id.')
        if patient_map and r['study_id'] not in patient_map:
            raise ValueError(f'Missing patient mapping for {r["study_id"]}')
        if any(r[c] not in ('0', '1') for c in CLASSES):
            raise ValueError(f'Non-binary label for {r["img_id"]}')
        if not (args.images_dir / r['img_id']).is_file():
            raise FileNotFoundError(args.images_dir / r['img_id'])
        group = patient_map[r['study_id']] if args.patient_map else r['study_id']
        groups[group].append(r)
    partitions, balance_score = partition_groups(groups, args.seed, args.strategy, args.candidates)
    records = {}
    stats = {}
    for split, group_ids in partitions.items():
        selected = [r for g in group_ids for r in groups[g]]
        positive = {c: sum(int(r[c]) for r in selected) for c in CLASSES}
        if not selected or any(v == 0 or v == len(selected) for v in positive.values()):
            raise ValueError(f'{split} lacks positive or negative examples: {positive}. '
                             'Use a documented label-stratified group protocol before training.')
        records[split] = selected
        stats[split] = {'groups': len(group_ids), 'studies': len({r['study_id'] for r in selected}),
                        'images': len(selected), 'positive_images': positive,
                        'positive_groups': {c: sum(any(int(r[c]) for r in groups[g]) for g in group_ids)
                                            for c in CLASSES}}
    overlap = {}
    names = list(partitions)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            overlap[f'{a}-{b}'] = {
                'groups': len(set(partitions[a]) & set(partitions[b])),
                'studies': len({r['study_id'] for r in records[a]} & {r['study_id'] for r in records[b]}),
                'images': len({r['img_id'] for r in records[a]} & {r['img_id'] for r in records[b]}),
            }
    assert sum(len(v) for v in records.values()) == len(rows)
    assert all(n == 0 for counts in overlap.values() for n in counts.values())
    patient_verified = bool(args.patient_map or args.confirm_study_is_patient)
    report = {'seed': args.seed, 'ratios': [0.7, 0.1, 0.2],
              'assignment_strategy': args.strategy,
              'candidate_count': args.candidates if args.strategy == 'balanced' else 1,
              'balance_score': balance_score,
              'balance_objective': 'sum of squared relative deviations from split-ratio targets '
                  'for image count, positive-image counts, and positive-group counts',
              'coverage_constraints': 'at least 10 positive and 10 negative images and '
                  '2 positive groups per class in each split' if args.strategy == 'balanced' else None,
              'group_type': 'patient' if patient_verified else 'study',
              'patient_independence_verified': patient_verified,
              'verification_basis': 'provided patient mapping' if args.patient_map else
                  ('user confirmation of one study per patient' if args.confirm_study_is_patient else 'not verified'),
              'source_csv': str(args.csv.resolve()),
              'source_sha256': hashlib.sha256(args.csv.read_bytes()).hexdigest(),
              'patient_map': str(args.patient_map.resolve()) if args.patient_map else None,
              'patient_map_sha256': hashlib.sha256(args.patient_map.read_bytes()).hexdigest() if args.patient_map else None,
              'split_statistics': stats, 'overlap': overlap,
              'group_assignments': {k: sorted(v) for k, v in partitions.items()}}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, selected in records.items():
        data_list = [{'img_path': str(args.images_dir.resolve() / r['img_id']),
                      'gt_score': [float(r[c]) for c in CLASSES], 'study_id': r['study_id'],
                      'group_id': patient_map[r['study_id']] if args.patient_map else r['study_id']}
                     for r in selected]
        with (args.out_dir / f'{split}_list.pkl').open('wb') as f:
            pickle.dump({'metainfo': {'classes': CLASSES}, 'data_list': data_list}, f)
    with (args.out_dir / 'split_manifest.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'group_id', 'study_id', 'img_id'] + CLASSES)
        for split, selected in records.items():
            for r in selected:
                writer.writerow([split, patient_map[r['study_id']] if args.patient_map else r['study_id'],
                                 r['study_id'], r['img_id']] + [r[c] for c in CLASSES])
    (args.out_dir / 'split_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'group_assignments'}, indent=2))


if __name__ == '__main__':
    main()
