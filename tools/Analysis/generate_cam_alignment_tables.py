import argparse
import csv
import json
from pathlib import Path


DATASETS = [
    ("FFA", "LID-FFA"),
    ("Endo", "Endo"),
    ("ODIR", "ODIR-5K"),
]

METHODS = [
    ("baseline", "Q2L"),
    ("all_loss", "Q2L-TDE"),
]

ALL_METRICS = [
    "camalign_pearson",
    "camalign_cosine",
    "camalign_dice@10",
    "camalign_iou@10",
    "camalign_dice@20",
    "camalign_iou@20",
    "camalign_dice@30",
    "camalign_iou@30",
    "cls_proto_cosine",
    "cls_tde_cosine",
    "proto_tde_cosine",
]

COMPACT_METRICS = [
    ("camalign_pearson", "CAM-Align Pearson"),
    ("camalign_cosine", "CAM-Align Cosine"),
    ("camalign_dice@20", "CAM-Align Dice@20"),
    ("camalign_iou@20", "CAM-Align IoU@20"),
]

PAIRWISE_METRICS = [
    ("cls_proto_cosine", "Cls-Proto"),
    ("cls_tde_cosine", "Cls-TDE"),
    ("proto_tde_cosine", "Proto-TDE"),
    ("camalign_cosine", "CAM-Align"),
]


def load_summary(summary_path):
    data = json.loads(Path(summary_path).read_text())
    overall = {item["metric"]: item for item in data["overall"]}
    return data, overall


def collect_rows(root):
    root = Path(root)
    rows = []
    for dataset_dir, dataset_name in DATASETS:
        for method_dir, method_name in METHODS:
            summary_path = (
                root
                / dataset_dir
                / "Component_Ablation"
                / method_dir
                / "fold0"
                / "vis_cam"
                / "cam_alignment_positive_summary.json"
            )
            if not summary_path.exists():
                raise FileNotFoundError(f"Missing CAM alignment summary: {summary_path}")

            data, overall = load_summary(summary_path)
            row = {
                "dataset": dataset_name,
                "method": method_name,
                "num_positive_rows": data.get("num_positive_rows"),
            }
            for metric in ALL_METRICS:
                if metric not in overall:
                    raise KeyError(f"Metric {metric} not found in {summary_path}")
                row[metric] = overall[metric]["mean"]
                row[f"{metric}_std"] = overall[metric]["std"]
            rows.append(row)
    return rows


def write_csv(rows, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def latex_value(value, is_best):
    text = f"{value:.3f}"
    if is_best:
        text = r"\mathbf{" + text + "}"
    return f"${text}$"


def render_latex_table(rows, metric_specs, caption, label):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\resizebox{\columnwidth}{!}{",
        r"\begin{tabular}{l|l|" + "c" * len(metric_specs) + "}",
        r"\hline",
        r"\textbf{Dataset} & \textbf{Method} & "
        + " & ".join(rf"\textbf{{{name}}}" for _, name in metric_specs)
        + r" \\",
        r"\hline",
    ]

    for _, dataset_name in DATASETS:
        dataset_rows = [row for row in rows if row["dataset"] == dataset_name]
        max_by_metric = {
            metric: max(row[metric] for row in dataset_rows)
            for metric, _ in metric_specs
        }
        for idx, row in enumerate(dataset_rows):
            dataset_cell = rf"\multirow{{{len(dataset_rows)}}}{{*}}{{{dataset_name}}}" if idx == 0 else ""
            values = [
                latex_value(row[metric], abs(row[metric] - max_by_metric[metric]) < 1e-12)
                for metric, _ in metric_specs
            ]
            lines.append(f"{dataset_cell} & {row['method']} & " + " & ".join(values) + r" \\")
        lines.append(r"\hline")

    lines.extend([
        r"\end{tabular}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate CSV and LaTeX tables for positive-sample CAM alignment summaries."
    )
    parser.add_argument(
        "--root",
        default="work_dirs/work_dirs_Prototype_TDE_0602",
        help="Root folder containing FFA/Endo/ODIR Component_Ablation results.",
    )
    parser.add_argument(
        "--out-dir",
        default="work_dirs/work_dirs_Prototype_TDE_0602/Figures",
        help="Output directory for generated CSV and .tex files.",
    )
    args = parser.parse_args()

    rows = collect_rows(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "cam_alignment_table_values.csv"
    compact_path = out_dir / "cam_alignment_table_compact.tex"
    pairwise_path = out_dir / "cam_alignment_table_pairwise.tex"

    write_csv(rows, csv_path)

    compact_caption = (
        "Quantitative Grad-CAM alignment comparison between Q2L and Q2L-TDE. "
        "Higher values indicate stronger spatial consistency among classification CAM, "
        "prototype-similarity CAM, and TDE-response CAM."
    )
    compact_tex = render_latex_table(
        rows,
        COMPACT_METRICS,
        compact_caption,
        "tab:cam_alignment",
    )
    compact_path.write_text(compact_tex)

    pairwise_caption = (
        "Pairwise Grad-CAM cosine alignment. Cls-Proto measures consistency between "
        "classification evidence and prototype semantics; Cls-TDE measures consistency "
        "between classification evidence and causal response; Proto-TDE measures consistency "
        "between prototype semantics and causal response. CAM-Align is the average of the "
        "three pairwise alignments."
    )
    pairwise_tex = render_latex_table(
        rows,
        PAIRWISE_METRICS,
        pairwise_caption,
        "tab:cam_pairwise_alignment",
    )
    pairwise_path.write_text(pairwise_tex)

    print(f"Saved: {csv_path}")
    print(f"Saved: {compact_path}")
    print(f"Saved: {pairwise_path}")


if __name__ == "__main__":
    main()
