import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DATASET_CLASSES = {
    'FFA': ['L', 'TP', 'ST', 'SD', 'NP', 'VA'],
    'LID-FFA': ['L', 'TP', 'ST', 'SD', 'NP', 'VA'],
    'Endo': ['U', 'E', 'P', 'T'],
    'ODIR': ['D', 'G', 'C', 'A', 'H', 'M', 'O'],
    'ODIR-5K': ['D', 'G', 'C', 'A', 'H', 'M', 'O'],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Interactively select one positive Grad-CAM sample per disease.')
    parser.add_argument('--dataset', required=True, choices=sorted(DATASET_CLASSES),
                        help='Dataset name used for class labels.')
    parser.add_argument('--q2l-vis-dir', required=True, type=Path,
                        help='Baseline Q2L vis_cam directory.')
    parser.add_argument('--tde-vis-dir', required=True, type=Path,
                        help='Q2L-TDE vis_cam directory.')
    parser.add_argument('--out', type=Path,
                        default=Path('work_dirs/work_dirs_Prototype_TDE_0602/Figures/cam_selected_samples.json'),
                        help='Output JSON path for accepted samples.')
    parser.add_argument('--start-class', type=int, default=0,
                        help='Class index to start from.')
    parser.add_argument('--sort-by', default='camalign_cosine',
                        help='Column in Q2L-TDE cam_alignment_positive_per_class.csv used to sort candidates.')
    parser.add_argument('--ascending', action='store_true',
                        help='Sort candidates ascending instead of descending.')
    parser.add_argument('--max-candidates', type=int, default=0,
                        help='Maximum candidates to browse per class. 0 means all.')
    parser.add_argument('--contact-sheet', action='store_true',
                        help='Generate non-interactive candidate contact sheets and exit.')
    parser.add_argument('--contact-cols', type=int, default=2,
                        help='Number of columns in contact sheets.')
    parser.add_argument('--thumb-width', type=int, default=900,
                        help='Width used for each model CAM thumbnail in contact sheets.')
    return parser.parse_args()


def load_positive_rows(vis_dir):
    csv_path = vis_dir / 'cam_alignment_positive_per_class.csv'
    if not csv_path.is_file():
        raise FileNotFoundError(f'Missing positive CAM summary: {csv_path}')
    rows = pd.read_csv(csv_path)
    required = {'image_name', 'class_idx'}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f'{csv_path} is missing columns: {sorted(missing)}')
    return rows


def find_cam_image(vis_dir, image_name):
    matches = sorted(vis_dir.glob(f'*_{image_name}'))
    if not matches:
        matches = sorted(vis_dir.glob(f'*{image_name}'))
    return matches[0] if matches else None


def make_candidate_rows(tde_rows, class_idx, sort_by, ascending, max_candidates):
    rows = tde_rows[tde_rows['class_idx'].astype(int) == int(class_idx)].copy()
    if sort_by in rows.columns:
        rows = rows.sort_values(sort_by, ascending=ascending, na_position='last')
    else:
        rows = rows.sort_values('image_name')
    if max_candidates > 0:
        rows = rows.head(max_candidates)
    return rows.to_dict(orient='records')


def show_pair(q2l_path, tde_path, dataset, class_idx, class_name, row, candidate_idx, total):
    q2l_img = mpimg.imread(q2l_path)
    tde_img = mpimg.imread(tde_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(q2l_img)
    axes[0].set_title('Q2L', fontsize=13)
    axes[1].imshow(tde_img)
    axes[1].set_title('Q2L-TDE', fontsize=13)
    for ax in axes:
        ax.axis('off')

    score_bits = []
    for key in ('pred_score', 'camalign_pearson', 'camalign_cosine', 'camalign_dice@20', 'camalign_iou@20'):
        if key in row and pd.notna(row[key]):
            score_bits.append(f'{key}={float(row[key]):.3f}')
    subtitle = ' | '.join(score_bits)
    fig.suptitle(
        f'{dataset} class {class_idx}: {class_name}  '
        f'candidate {candidate_idx + 1}/{total}\n'
        f'{row.get("image_name", "")}\n'
        f'{subtitle}\n'
        'Keys: y=accept, n/space=next, b=previous, q=quit',
        fontsize=11,
    )
    fig.tight_layout()

    pressed = {'key': None}

    def on_key(event):
        pressed['key'] = event.key
        plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    return pressed['key']


def write_outputs(out_path, selections):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        json.dump(selections, f, indent=2)

    csv_path = out_path.with_suffix('.csv')
    fieldnames = [
        'dataset', 'class_idx', 'class_name', 'image_name', 'image_path',
        'q2l_cam_path', 'tde_cam_path', 'pred_score',
        'camalign_pearson', 'camalign_cosine', 'camalign_dice@20', 'camalign_iou@20',
    ]
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in selections:
            writer.writerow({key: item.get(key, '') for key in fieldnames})
    print(f'Saved selections: {out_path}')
    print(f'Saved selections: {csv_path}')


def resize_to_width(image, width):
    ratio = width / image.width
    height = max(1, int(round(image.height * ratio)))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def make_pair_tile(q2l_path, tde_path, header, thumb_width=900, pad=12):
    q2l_img = resize_to_width(Image.open(q2l_path).convert('RGB'), thumb_width)
    tde_img = resize_to_width(Image.open(tde_path).convert('RGB'), thumb_width)
    font = ImageFont.load_default()
    header_h = 44
    label_h = 22
    width = thumb_width
    height = header_h + label_h + q2l_img.height + label_h + tde_img.height + pad * 3
    tile = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(tile)
    draw.text((pad, pad), header, fill='black', font=font)
    y = header_h
    draw.text((pad, y), 'Q2L', fill='black', font=font)
    y += label_h
    tile.paste(q2l_img, (0, y))
    y += q2l_img.height + pad
    draw.text((pad, y), 'Q2L-TDE', fill='black', font=font)
    y += label_h
    tile.paste(tde_img, (0, y))
    return tile


def generate_contact_sheets(args):
    class_names = DATASET_CLASSES[args.dataset]
    tde_rows = load_positive_rows(args.tde_vis_dir)
    out_dir = args.out.with_suffix('')
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for class_idx, class_name in enumerate(class_names):
        if class_idx < args.start_class:
            continue
        candidates = make_candidate_rows(
            tde_rows, class_idx, args.sort_by, args.ascending,
            args.max_candidates if args.max_candidates > 0 else 12)
        tiles = []
        for cand_idx, row in enumerate(candidates):
            image_name = row['image_name']
            q2l_cam = find_cam_image(args.q2l_vis_dir, image_name)
            tde_cam = find_cam_image(args.tde_vis_dir, image_name)
            if q2l_cam is None or tde_cam is None:
                continue
            metrics = []
            for key in ('pred_score', 'camalign_pearson', 'camalign_cosine', 'camalign_dice@20', 'camalign_iou@20'):
                if key in row and pd.notna(row[key]):
                    metrics.append(f'{key}={float(row[key]):.3f}')
            header = f'#{cand_idx:02d} {args.dataset} {class_idx}:{class_name}  {image_name}  ' + '  '.join(metrics[:3])
            tiles.append(make_pair_tile(q2l_cam, tde_cam, header, thumb_width=args.thumb_width))
            manifest = {
                'dataset': args.dataset,
                'class_idx': class_idx,
                'class_name': class_name,
                'candidate_idx': cand_idx,
                'image_name': image_name,
                'image_path': row.get('image_path', ''),
                'q2l_cam_path': str(q2l_cam),
                'tde_cam_path': str(tde_cam),
            }
            for key in ('pred_score', 'camalign_pearson', 'camalign_cosine', 'camalign_dice@20', 'camalign_iou@20'):
                if key in row and pd.notna(row[key]):
                    manifest[key] = float(row[key])
            manifest_rows.append(manifest)

        if not tiles:
            print(f'No paired candidates for class {class_idx}: {class_name}')
            continue

        cols = max(1, args.contact_cols)
        rows = int(math.ceil(len(tiles) / cols))
        cell_w = max(tile.width for tile in tiles)
        cell_h = max(tile.height for tile in tiles)
        sheet = Image.new('RGB', (cols * cell_w, rows * cell_h), 'white')
        for tile_idx, tile in enumerate(tiles):
            x = (tile_idx % cols) * cell_w
            y = (tile_idx // cols) * cell_h
            sheet.paste(tile, (x, y))
        out_path = out_dir / f'{args.dataset}_class{class_idx}_{class_name}_candidates.png'
        sheet.save(out_path)
        print(f'Saved contact sheet: {out_path}')

    manifest_path = out_dir / 'contact_sheet_manifest.csv'
    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
        print(f'Saved manifest: {manifest_path}')


def main():
    args = parse_args()
    if args.contact_sheet:
        generate_contact_sheets(args)
        return

    class_names = DATASET_CLASSES[args.dataset]
    tde_rows = load_positive_rows(args.tde_vis_dir)
    _ = load_positive_rows(args.q2l_vis_dir)

    selections = []
    if args.out.is_file():
        with args.out.open() as f:
            selections = json.load(f)
        print(f'Loaded existing selections: {args.out}')

    selected_classes = {int(item['class_idx']) for item in selections}

    for class_idx in range(args.start_class, len(class_names)):
        if class_idx in selected_classes:
            print(f'Skip already selected class {class_idx}: {class_names[class_idx]}')
            continue

        candidates = make_candidate_rows(
            tde_rows, class_idx, args.sort_by, args.ascending, args.max_candidates)
        if not candidates:
            print(f'No positive candidates for class {class_idx}: {class_names[class_idx]}')
            continue

        print(f'\nClass {class_idx}: {class_names[class_idx]}  candidates={len(candidates)}')
        idx = 0
        while 0 <= idx < len(candidates):
            row = candidates[idx]
            image_name = row['image_name']
            q2l_cam = find_cam_image(args.q2l_vis_dir, image_name)
            tde_cam = find_cam_image(args.tde_vis_dir, image_name)
            if q2l_cam is None or tde_cam is None:
                print(f'  Missing paired CAM for {image_name}: q2l={q2l_cam}, tde={tde_cam}')
                idx += 1
                continue

            key = show_pair(
                q2l_cam, tde_cam, args.dataset, class_idx, class_names[class_idx],
                row, idx, len(candidates))
            if key in ('y', 'Y', 'enter'):
                item = {
                    'dataset': args.dataset,
                    'class_idx': int(class_idx),
                    'class_name': class_names[class_idx],
                    'image_name': image_name,
                    'image_path': row.get('image_path', ''),
                    'q2l_cam_path': str(q2l_cam),
                    'tde_cam_path': str(tde_cam),
                }
                for metric in (
                    'pred_score', 'camalign_pearson', 'camalign_cosine',
                    'camalign_dice@20', 'camalign_iou@20',
                ):
                    if metric in row and pd.notna(row[metric]):
                        item[metric] = float(row[metric])
                selections.append(item)
                selected_classes.add(class_idx)
                write_outputs(args.out, selections)
                break
            if key in ('b', 'B', 'left'):
                idx = max(0, idx - 1)
            elif key in ('q', 'Q', 'escape'):
                write_outputs(args.out, selections)
                return
            else:
                idx += 1

        if class_idx not in selected_classes:
            print(f'No sample accepted for class {class_idx}: {class_names[class_idx]}')

    write_outputs(args.out, selections)
    print('Done.')


if __name__ == '__main__':
    main()
