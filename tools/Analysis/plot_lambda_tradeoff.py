import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


DATASET_FILES = {
    "FFA": (
        "LID-FFA",
        [
            "FFA/FFA_Lambda_Ablation_margin0.5.md",
            "FFA/FFA_Lambda_Ablation_margin0.5_new.md",
        ],
        "FFA/FFA_Component_Ablation.md",
        "FFA/FFA_Margin_Ablation.md",
    ),
    "Endo": (
        "Endo",
        "Endo/Endo_Lambda_Ablation.md",
        "Endo/Endo_Component_Ablation.md",
        "Endo/Endo_Margin_Ablation.md",
    ),
    "ODIR": (
        "ODIR-5K",
        "ODIR/ODIR_Lambda_Ablation_margin0.5.md",
        "ODIR/ODIR_Component_Ablation.md",
        "ODIR/ODIR_Margin_Ablation.md",
    ),
}


def parse_value(cell):
    cell = cell.strip()
    if not cell:
        return np.nan
    match = re.match(r"([-+]?\d*\.?\d+)", cell)
    if not match:
        raise ValueError(f"Cannot parse numeric value from: {cell}")
    return float(match.group(1))


def read_md_tables(path):
    lines = Path(path).read_text().splitlines()
    tables = {}
    current_title = None
    current_lines = []

    def flush():
        if current_title and current_lines:
            headers = [h.strip() for h in current_lines[0].strip("|").split("|")]
            row_labels = []
            values = []
            for line in current_lines[2:]:
                cells = [c.strip() for c in line.strip("|").split("|")]
                if len(cells) != len(headers):
                    continue
                row_labels.append(float(cells[0]))
                values.append([parse_value(cell) for cell in cells[1:]])
            tables[current_title] = {
                "cols": [float(h) for h in headers[1:]],
                "rows": row_labels,
                "values": np.array(values, dtype=float),
            }

    for line in lines:
        if line.startswith("### "):
            flush()
            current_title = line.replace("### ", "").strip()
            current_lines = []
        elif line.strip().startswith("|"):
            current_lines.append(line)
    flush()
    return tables


def read_component_baseline(path):
    lines = Path(path).read_text().splitlines()
    table_lines = [line for line in lines if line.strip().startswith("|")]
    headers = [h.strip() for h in table_lines[0].strip("|").split("|")]
    rows = {}
    for line in table_lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == len(headers):
            rows[cells[0]] = dict(zip(headers[1:], cells[1:]))
    return {
        "mauc": parse_value(rows["mean mAUC"]["baseline"]),
        "mauc_std": parse_std(rows["mean mAUC"]["baseline"]),
        "or": parse_value(rows["Overall OR Median"]["baseline"]),
    }


def read_component_variant(path, variant):
    lines = Path(path).read_text().splitlines()
    table_lines = [line for line in lines if line.strip().startswith("|")]
    headers = [h.strip() for h in table_lines[0].strip("|").split("|")]
    rows = {}
    for line in table_lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == len(headers):
            rows[cells[0]] = dict(zip(headers[1:], cells[1:]))
    if variant not in rows["mean mAUC"]:
        raise ValueError(f"{path} does not contain component variant {variant}")
    return {
        "mauc": parse_value(rows["mean mAUC"][variant]),
        "mauc_std": parse_std(rows["mean mAUC"][variant]),
        "or": parse_value(rows["Overall OR Median"][variant]),
        "tde_neg": parse_value(rows["Overall TDE Negative"][variant]),
        "tde_pos": parse_value(rows["Overall TDE Positive"][variant]),
    }


def read_margin_full_model(path, margin="margin0.5"):
    lines = Path(path).read_text().splitlines()
    table_lines = [line for line in lines if line.strip().startswith("|")]
    headers = [h.strip() for h in table_lines[0].strip("|").split("|")]
    if margin not in headers:
        raise ValueError(f"{path} does not contain {margin}")
    margin_idx = headers.index(margin)
    rows = {}
    for line in table_lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == len(headers):
            rows[cells[0]] = cells[margin_idx]
    return {
        "mauc": parse_value(rows["mean mAUC"]),
        "mauc_std": parse_std(rows["mean mAUC"]),
        "or": parse_value(rows["Overall OR Median"]),
        "tde_neg": parse_value(rows["Overall TDE Negative"]),
        "tde_pos": parse_value(rows["Overall TDE Positive"]),
    }


def parse_std(cell):
    cell = cell.strip()
    match = re.match(r"[-+]?\d*\.?\d+\s*[±+]\s*([-+]?\d*\.?\d+)", cell)
    return float(match.group(1)) if match else 0.0


def heatmap(
    ax,
    values,
    row_labels,
    col_labels,
    title,
    cmap,
    vlim=None,
    fmt="{:+.2f}",
    annotations=None,
    outline_mask=None,
):
    if vlim is None:
        max_abs = np.nanmax(np.abs(values))
        vlim = max_abs if max_abs > 0 else 1.0
    im = ax.imshow(values, cmap=cmap, vmin=-vlim, vmax=vlim, aspect="equal")
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xlabel(r"$\lambda_1$", fontsize=9)
    ax.set_ylabel(r"$\lambda_2$", fontsize=9)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels([f"{v:.2f}" for v in col_labels], fontsize=8)
    ax.set_yticklabels([f"{v:.2f}" for v in row_labels], fontsize=8)
    ax.tick_params(length=0)

    threshold = vlim * 0.55
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if abs(values[i, j]) > threshold else "#333333"
            text = annotations[i][j] if annotations is not None else fmt.format(values[i, j])
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)
            if outline_mask is not None and outline_mask[i, j]:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#111111",
                        linewidth=1.4,
                    )
                )
    return im


def assert_same_grid(ref_table, next_table, path):
    if ref_table["rows"] != next_table["rows"] or ref_table["cols"] != next_table["cols"]:
        raise ValueError(f"Lambda grid mismatch in {path}")


def merge_lambda_tables(root, dataset, lambda_rel, baseline):
    paths = lambda_rel if isinstance(lambda_rel, (list, tuple)) else [lambda_rel]
    parsed = []
    for path in paths:
        tables = read_md_tables(root / path)
        parsed.append(
            {
                "path": path,
                "mauc": tables[f"{dataset} - mean mAUC"],
                "or": tables[f"{dataset} - Overall OR Median"],
            }
        )

    ref_mauc = parsed[0]["mauc"]
    ref_or = parsed[0]["or"]
    for item in parsed[1:]:
        assert_same_grid(ref_mauc, item["mauc"], item["path"])
        assert_same_grid(ref_or, item["or"], item["path"])

    if len(parsed) == 1:
        return ref_mauc, ref_or, np.zeros_like(ref_mauc["values"], dtype=int), paths

    mauc_stack = np.stack([item["mauc"]["values"] for item in parsed], axis=0)
    or_stack = np.stack([item["or"]["values"] for item in parsed], axis=0)

    delta_mauc_pct = (mauc_stack - baseline["mauc"]) * 100.0
    comparable_band_pct = baseline["mauc_std"] * 100.0
    comparable = np.abs(delta_mauc_pct) <= comparable_band_pct

    chosen = np.zeros_like(ref_mauc["values"], dtype=int)
    for i in range(ref_mauc["values"].shape[0]):
        for j in range(ref_mauc["values"].shape[1]):
            valid = np.where(comparable[:, i, j])[0]
            if len(valid) > 0:
                best_local = valid[np.nanargmax(or_stack[valid, i, j])]
            else:
                best_local = int(np.nanargmax(mauc_stack[:, i, j]))
            chosen[i, j] = best_local

    merged_mauc = {
        "rows": ref_mauc["rows"],
        "cols": ref_mauc["cols"],
        "values": np.take_along_axis(mauc_stack, chosen[None, :, :], axis=0)[0],
    }
    merged_or = {
        "rows": ref_or["rows"],
        "cols": ref_or["cols"],
        "values": np.take_along_axis(or_stack, chosen[None, :, :], axis=0)[0],
    }
    return merged_mauc, merged_or, chosen, paths


def write_selection_map(out_prefix, row_labels, col_labels, chosen, paths, full_model_source=None):
    if len(paths) <= 1:
        return
    out_path = Path(out_prefix).with_name(Path(out_prefix).name + "_selected_sources.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("lambda_2\\lambda_1," + ",".join(f"{v:.2f}" for v in col_labels) + "\n")
        for i, row_value in enumerate(row_labels):
            sources = [Path(paths[int(chosen[i, j])]).stem for j in range(len(col_labels))]
            if full_model_source and np.isclose(float(row_value), 1.0):
                for j, col_value in enumerate(col_labels):
                    if np.isclose(float(col_value), 1.0):
                        sources[j] = full_model_source
            f.write(f"{row_value:.2f}," + ",".join(sources) + "\n")
    print(f"Saved: {out_path}")


def override_full_model_cell(mauc_table, or_table, full_model):
    rows = np.array(mauc_table["rows"], dtype=float)
    cols = np.array(mauc_table["cols"], dtype=float)
    row_matches = np.where(np.isclose(rows, 1.0))[0]
    col_matches = np.where(np.isclose(cols, 1.0))[0]
    if len(row_matches) == 0 or len(col_matches) == 0:
        return
    i = int(row_matches[0])
    j = int(col_matches[0])
    mauc_table["values"][i, j] = full_model["mauc"]
    or_table["values"][i, j] = full_model["or"]


def plot_lambda_tradeoff(root, dataset, out_prefix, mauc_mode):
    root = Path(root)
    title, lambda_rel, component_rel, margin_rel = DATASET_FILES[dataset]
    baseline = read_component_baseline(root / component_rel)
    if dataset == "FFA":
        full_model = read_component_variant(root / component_rel, "all_loss")
        full_model_source = f"{Path(component_rel).stem}_all_loss"
    else:
        full_model = read_margin_full_model(root / margin_rel)
        full_model_source = f"{Path(margin_rel).stem}_margin0.5"

    mauc_table, or_table, chosen, source_paths = merge_lambda_tables(root, dataset, lambda_rel, baseline)
    override_full_model_cell(mauc_table, or_table, full_model)

    delta_mauc_pct = (mauc_table["values"] - baseline["mauc"]) * 100.0
    delta_or = or_table["values"] - baseline["or"]
    comparable_band_pct = baseline["mauc_std"] * 100.0
    non_comparable_mask = np.abs(delta_mauc_pct) > comparable_band_pct

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    if mauc_mode == "delta":
        mauc_values = delta_mauc_pct
        mauc_title = rf"$\Delta$ mAUC (%) vs Q2L ({baseline['mauc'] * 100:.2f}%)"
        mauc_cmap = "RdYlGn"
        mauc_vlim = None
        mauc_annotations = None
    elif mauc_mode == "comparable":
        mauc_values = np.zeros_like(delta_mauc_pct)
        mauc_values[delta_mauc_pct < -comparable_band_pct] = -1
        mauc_values[delta_mauc_pct > comparable_band_pct] = 1
        mauc_title = rf"$\Delta$ mAUC (%)"
        mauc_cmap = ListedColormap(["#d73027", "#f2f2f2", "#1a9850"])
        mauc_vlim = 1.5
        mauc_annotations = []
        for row in delta_mauc_pct:
            ann_row = []
            for value in row:
                ann_row.append(f"{value:+.2f}")
            mauc_annotations.append(ann_row)
    else:
        raise ValueError(f"Unsupported mAUC mode: {mauc_mode}")

    im0 = heatmap(
        axes[0],
        mauc_values,
        mauc_table["rows"],
        mauc_table["cols"],
        mauc_title,
        mauc_cmap,
        vlim=mauc_vlim,
        fmt="{:+.2f}",
        annotations=mauc_annotations,
        outline_mask=non_comparable_mask if mauc_mode == "comparable" else None,
    )
    im1 = heatmap(
        axes[1],
        delta_or,
        or_table["rows"],
        or_table["cols"],
        rf"$\Delta$ Median OR vs Q2L ({baseline['or']:.2f})",
        "RdYlGn",
        fmt="{:+.2f}",
    )
    cbar0 = fig.colorbar(
        im0,
        ax=axes[0],
        fraction=0.045,
        pad=0.02,
        extend="both" if mauc_mode == "comparable" else "neither",
    )
    if mauc_mode == "comparable":
        cbar0.set_ticks([-1, 0, 1])
        cbar0.set_ticklabels([r"$<-\sigma$", r"$|\Delta|\leq\sigma$", r"$>+\sigma$"])
    fig.colorbar(im1, ax=axes[1], fraction=0.045, pad=0.02)

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {out_prefix.with_suffix('.png')}")
    print(f"Saved: {out_prefix.with_suffix('.pdf')}")
    write_selection_map(
        out_prefix,
        mauc_table["rows"],
        mauc_table["cols"],
        chosen,
        source_paths,
        full_model_source=full_model_source,
    )


def main():
    parser = argparse.ArgumentParser(description="Plot lambda trade-off heatmaps for mAUC and Median OR.")
    parser.add_argument("--root", default="work_dirs/work_dirs_Prototype_TDE_0602")
    parser.add_argument("--dataset", choices=sorted(DATASET_FILES), default="ODIR")
    parser.add_argument("--out", default=None, help="Output prefix. Defaults to Figures/lambda_tradeoff_<dataset>.")
    parser.add_argument(
        "--mauc-mode",
        choices=["delta", "comparable"],
        default="comparable",
        help="Show raw delta mAUC or status relative to the baseline standard deviation.",
    )
    args = parser.parse_args()

    out = args.out
    if out is None:
        out = f"work_dirs/work_dirs_Prototype_TDE_0602/Figures/lambda_ablation/lambda_tradeoff_{args.dataset.lower()}"
    plot_lambda_tradeoff(args.root, args.dataset, out, args.mauc_mode)


if __name__ == "__main__":
    main()
