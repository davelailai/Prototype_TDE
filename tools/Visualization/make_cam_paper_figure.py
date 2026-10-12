import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


SIGNAL_NAMES = ['Classification', 'Prototype', 'TDE']
DEFAULT_COL_TITLES = [
    'Disease', 'Image',
    'Q2L\nCls.', 'Q2L\nProto.', 'Q2L\nTDE',
    'Q2L-TDE\nCls.', 'Q2L-TDE\nProto.', 'Q2L-TDE\nTDE',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build a paper-ready CAM comparison figure from selected samples.')
    parser.add_argument('--selection', type=Path, required=True,
                        help='Selection JSON produced by select_cam_samples.py.')
    parser.add_argument('--out', type=Path, required=True,
                        help='Output path without extension or with .png/.pdf.')
    parser.add_argument('--dpi', type=int, default=350)
    parser.add_argument('--fig-width', type=float, default=14.0)
    parser.add_argument('--row-height', type=float, default=1.75)
    parser.add_argument('--image-size', type=int, default=256,
                        help='Square size for each shown crop.')
    parser.add_argument('--inner-crop-ratio', type=float, default=0.04,
                        help='Remove this fraction from each CAM crop border before resizing.')
    return parser.parse_args()


def load_selection(path):
    with path.open() as f:
        rows = json.load(f)
    rows = sorted(rows, key=lambda item: int(item['class_idx']))
    return rows


def crop_inner(img, ratio):
    if ratio <= 0:
        return img
    w, h = img.size
    dx = int(round(w * ratio))
    dy = int(round(h * ratio))
    if dx * 2 >= w or dy * 2 >= h:
        return img
    return img.crop((dx, dy, w - dx, h - dy))


def resize_square(img, size, inner_crop_ratio=0.0):
    img = crop_inner(img, inner_crop_ratio)
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return img


def crop_target_cam_grid(cam_path, class_idx, image_size, inner_crop_ratio=0.0):
    """Crop original and target-class CAMs from saved Grad-CAM montage.

    The source montage layout is:
      vertical blocks: classification, prototype similarity, TDE response
      horizontal columns inside each block: original image + all class CAMs
    """
    img = Image.open(cam_path).convert('RGB')
    width, height = img.size
    num_classes = 6 if width % 7 == 0 else None
    if num_classes is None:
        raise ValueError(
            f'Cannot infer class columns from {cam_path}: image size={img.size}. '
            'Expected width divisible by 7 for FFA.')

    block_h = height // 3
    col_w = width // (num_classes + 1)
    original = img.crop((0, 0, col_w, block_h))
    target_col = int(class_idx) + 1
    cams = []
    for block_idx in range(3):
        y0 = block_idx * block_h
        x0 = target_col * col_w
        cams.append(img.crop((x0, y0, x0 + col_w, y0 + block_h)))
    return [resize_square(original, image_size, 0.0)] + [
        resize_square(cam, image_size, inner_crop_ratio) for cam in cams
    ]


def draw_image(ax, pil_img):
    ax.imshow(pil_img)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def build_figure(rows, out_path, dpi=350, fig_width=14.0, row_height=1.75,
                 image_size=256, inner_crop_ratio=0.04):
    n_rows = len(rows)
    fig_height = row_height * n_rows + 0.85
    width_ratios = [0.62, 1, 1, 1, 1, 1, 1, 1]
    fig, axes = plt.subplots(
        n_rows + 1, 8,
        figsize=(fig_width, fig_height),
        gridspec_kw={
            'height_ratios': [0.22] + [1] * n_rows,
            'width_ratios': width_ratios,
            'wspace': 0.035,
            'hspace': 0.075,
        },
    )

    for col_idx, title in enumerate(DEFAULT_COL_TITLES):
        ax = axes[0, col_idx]
        ax.axis('off')
        ax.text(0.5, 0.5, title, ha='center', va='center', fontsize=9, fontweight='bold')

    for row_idx, item in enumerate(rows, start=1):
        class_idx = int(item['class_idx'])
        class_name = item['class_name']
        q2l_crops = crop_target_cam_grid(
            item['q2l_cam_path'], class_idx, image_size, inner_crop_ratio)
        tde_crops = crop_target_cam_grid(
            item['tde_cam_path'], class_idx, image_size, inner_crop_ratio)

        label_ax = axes[row_idx, 0]
        label_ax.axis('off')
        label_ax.text(
            0.5, 0.5, class_name,
            ha='center', va='center',
            fontsize=10, fontweight='bold',
        )

        draw_image(axes[row_idx, 1], tde_crops[0])
        for offset, crop in enumerate(q2l_crops[1:]):
            draw_image(axes[row_idx, 2 + offset], crop)
        for offset, crop in enumerate(tde_crops[1:]):
            draw_image(axes[row_idx, 5 + offset], crop)

    for ax in axes.reshape(-1):
        ax.set_facecolor('white')

    fig.patch.set_facecolor('white')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in {'.png', '.pdf', '.jpg', '.jpeg'}:
        base = out_path.with_suffix('')
    else:
        base = out_path
    png_path = base.with_suffix('.png')
    pdf_path = base.with_suffix('.pdf')
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(f'Saved: {png_path}')
    print(f'Saved: {pdf_path}')


def main():
    args = parse_args()
    rows = load_selection(args.selection)
    build_figure(
        rows,
        args.out,
        dpi=args.dpi,
        fig_width=args.fig_width,
        row_height=args.row_height,
        image_size=args.image_size,
        inner_crop_ratio=args.inner_crop_ratio,
    )


if __name__ == '__main__':
    main()
