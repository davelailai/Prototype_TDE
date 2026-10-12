import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASETS = [
    ("FFA", "LID-FFA", "FFA_Margin_Ablation.md", "FFA_Component_Ablation.md"),
    ("Endo", "Endo", "Endo_Margin_Ablation.md", "Endo_Component_Ablation.md"),
    ("ODIR", "ODIR-5K", "ODIR_Margin_Ablation.md", "ODIR_Component_Ablation.md"),
]


def parse_mean_std(cell):
    cell = cell.strip()
    if not cell:
        return np.nan, np.nan
    match = re.match(r"([-+]?\d*\.?\d+)\s*[±+]\s*([-+]?\d*\.?\d+)", cell)
    if match:
        return float(match.group(1)), float(match.group(2))
    return float(cell), 0.0


def read_md_table(path):
    lines = Path(path).read_text().splitlines()
    table_lines = [line for line in lines if line.strip().startswith("|")]
    if len(table_lines) < 3:
        raise ValueError(f"No markdown table found in {path}")

    headers = [h.strip() for h in table_lines[0].strip("|").split("|")]
    rows = {}
    for line in table_lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        rows[cells[0]] = dict(zip(headers[1:], cells[1:]))
    return headers[1:], rows


def read_margin_summary(path):
    columns, rows = read_md_table(path)
    margins = [float(col.replace("margin", "")) for col in columns]

    mauc = [parse_mean_std(rows["mean mAUC"][col]) for col in columns]
    tde_pos = [parse_mean_std(rows["Overall TDE Positive"][col]) for col in columns]
    median_or = [parse_mean_std(rows["Overall OR Median"][col]) for col in columns]

    return {
        "margins": np.array(margins),
        "mauc_mean": np.array([v[0] for v in mauc]),
        "mauc_std": np.array([v[1] for v in mauc]),
        "tde_mean": np.array([v[0] for v in tde_pos]),
        "tde_std": np.array([v[1] for v in tde_pos]),
        "or_mean": np.array([v[0] for v in median_or]),
        "or_std": np.array([v[1] for v in median_or]),
    }


def read_baseline(component_path):
    _, rows = read_md_table(component_path)
    mauc_mean, _ = parse_mean_std(rows["mean mAUC"]["baseline"])
    tde_mean, _ = parse_mean_std(rows["Overall TDE Positive"]["baseline"])
    or_mean, _ = parse_mean_std(rows["Overall OR Median"]["baseline"])
    return mauc_mean, tde_mean, or_mean


def plot_tradeoff(root, out_prefix, causal_metric):
    root = Path(root)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.1), constrained_layout=True)
    if causal_metric == "or":
        causal_mean_key = "or_mean"
        causal_std_key = "or_std"
        causal_label = "Median OR"
        causal_baseline_label = "Median OR Baseline"
        causal_ylabel = "Median OR"
    elif causal_metric == "tde_pos":
        causal_mean_key = "tde_mean"
        causal_std_key = "tde_std"
        causal_label = r"Avg.TDE$_+$"
        causal_baseline_label = r"Avg.TDE$_+$ Baseline"
        causal_ylabel = r"Avg.TDE$_+$"
    else:
        raise ValueError(f"Unsupported causal metric: {causal_metric}")

    for idx, (folder, title, margin_file, component_file) in enumerate(DATASETS):
        ax = axes[idx]
        ax_causal = ax.twinx()

        margin_path = root / folder / margin_file
        component_path = root / folder / component_file
        data = read_margin_summary(margin_path)
        baseline_mauc, baseline_tde, baseline_or = read_baseline(component_path)

        x = data["margins"]
        mauc = data["mauc_mean"]
        mauc_std = data["mauc_std"]
        causal = data[causal_mean_key]
        causal_std = data[causal_std_key]
        baseline_causal = baseline_or if causal_metric == "or" else baseline_tde

        line_mauc = ax.plot(x, mauc, color="#1f77b4", marker="o", linewidth=1.6, label="mAUC")
        ax.fill_between(x, mauc - mauc_std, mauc + mauc_std, color="#1f77b4", alpha=0.15, linewidth=0)
        base_mauc = ax.axhline(
            baseline_mauc, color="#1f77b4", linestyle="--", linewidth=0.9, label="mAUC Baseline"
        )

        line_causal = ax_causal.plot(
            x, causal, color="#ff7f0e", marker="o", linewidth=1.6, label=causal_label
        )
        if np.any(causal_std > 0):
            ax_causal.fill_between(
                x, causal - causal_std, causal + causal_std, color="#ff7f0e", alpha=0.16, linewidth=0
            )
        base_causal = ax_causal.axhline(
            baseline_causal, color="#ff7f0e", linestyle="--", linewidth=0.9, label=causal_baseline_label
        )

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Margin", fontsize=8)
        ax.set_ylabel("mAUC", color="#1f77b4", fontsize=8)
        ax_causal.set_ylabel(causal_ylabel, color="#ff7f0e", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{value:.1f}" for value in x])
        ax.tick_params(axis="both", labelsize=7)
        ax_causal.tick_params(axis="y", labelsize=7)
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.text(
            0.96,
            0.08,
            f"({chr(ord('a') + idx)})",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="right",
        )

        if idx == 0:
            handles = [line_mauc[0], base_mauc, line_causal[0], base_causal]
            labels = [h.get_label() for h in handles]
            ax.legend(handles, labels, fontsize=7, loc="upper left", frameon=False)

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {out_prefix.with_suffix('.png')}")
    print(f"Saved: {out_prefix.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser(description="Plot margin trade-off between mAUC and causal trustworthiness.")
    parser.add_argument(
        "--root",
        default="/media/jbridge/Expansion/code/Prototype-TDE/work_dirs/work_dirs_Prototype_TDE_0602",
        help="Root folder containing FFA/Endo/ODIR summary markdown files.",
    )
    parser.add_argument(
        "--out",
        default="work_dirs/work_dirs_Prototype_TDE_0602/Figures/margin_ablation/margin_tradeoff",
        help="Output prefix. The script writes both .png and .pdf.",
    )
    parser.add_argument(
        "--causal-metric",
        choices=["or", "tde_pos"],
        default="tde_pos",
        help="Causal metric shown on the right y-axis.",
    )
    args = parser.parse_args()
    plot_tradeoff(args.root, args.out, args.causal_metric)


if __name__ == "__main__":
    main()
