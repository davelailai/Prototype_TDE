import os
import json
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===================== 配置区 =====================
root_dir = "work_dirs_MICCAI2026_0215/ODIR/Counterfactual_Top_mask_lpos0.5_lneg0.1"
q2l_mauc = None         # 手动指定 Q2L mAUC 基准；若为 None 则从 q2l_result_path 读取
q2l_or_median = None     # 手动指定 Q2L OR Median 基准；若为 None 则从 q2l_result_path 读取
q2l_result_path = 'work_dirs_MICCAI2026_0215/ODIR/Ablation_cf0.4_margin0.4_lpos0.1_lneg0.5/Q2L'
metric_key = "multi-class/mAUC_multiclass"
num_folds = 5
# ==================================================


def get_latest_json(fold_path):
    json_files = glob.glob(os.path.join(fold_path, "**", "*.json"), recursive=True)
    if not json_files:
        return None
    return sorted(json_files)[-1]


def read_metric(json_path, key):
    if json_path is None or not os.path.exists(json_path):
        return None
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get(key, None)


def get_exp_mean(exp_dir, key, n_folds):
    values = []
    for i in range(n_folds):
        fold_path = os.path.join(exp_dir, f"fold{i}")
        json_path = get_latest_json(fold_path)
        val = read_metric(json_path, key)
        if val is not None:
            values.append(val)
    if not values:
        return None
    return float(np.mean(values)) * 100  # 转为百分比


def read_or_median_from_tde(exp_dir, n_folds):
    """从 exp_dir/fold{i}/eval/tde_stats_epoch_0.md 读取 Overall OR Median，返回均值。"""
    values = []
    # 匹配格式：- **Overall OR Median:** 数值
    pattern = re.compile(
        r"-\s*\*\*Overall OR Median:\*\*\s*([\-\d.eE+]+)",
        re.IGNORECASE
    )
    for i in range(n_folds):
        md_path = os.path.join(exp_dir, f"fold{i}", "eval", "tde_stats_epoch_0.md")
        if not os.path.exists(md_path):
            continue
        with open(md_path, "r") as f:
            content = f.read()
        m = pattern.search(content)
        if m:
            try:
                val = float(m.group(1))
                if not np.isnan(val) and not np.isinf(val):
                    values.append(val)
            except ValueError:
                pass
    if not values:
        return None
    return float(np.mean(values))


# ---------- 读取 Q2L 基准 ----------
if q2l_mauc is None:
    if q2l_result_path is None:
        raise ValueError("请设置 q2l_mauc 或 q2l_result_path")
    q2l_mauc = get_exp_mean(q2l_result_path, metric_key, num_folds)  # 已 ×100
    if q2l_mauc is None:
        raise RuntimeError(f"无法从 {q2l_result_path} 读取 Q2L mAUC")
    print(f"Q2L mAUC  (mean {num_folds}-fold): {q2l_mauc:.2f}%")
else:
    q2l_mauc = q2l_mauc * 100  # 手动输入也统一转百分比
    print(f"Q2L mAUC  (manual): {q2l_mauc:.2f}%")

if q2l_or_median is None and q2l_result_path is not None:
    q2l_or_median = read_or_median_from_tde(q2l_result_path, num_folds)
    if q2l_or_median is not None:
        print(f"Q2L OR Median (mean {num_folds}-fold): {q2l_or_median:.4f}")
    else:
        print("Q2L OR Median: N/A（tde_stats_epoch_0.md 未找到）")
else:
    if q2l_or_median is not None:
        print(f"Q2L OR Median (manual): {q2l_or_median:.4f}")

# ---------- 扫描 counterfactual 实验目录 ----------
cf_set = set()
margin_set = set()
results_mauc = {}       # {(cf, margin): mean_mauc}
results_or   = {}       # {(cf, margin): mean_or_median}

for run in sorted(os.listdir(root_dir)):
    run_path = os.path.join(root_dir, run)
    if not os.path.isdir(run_path):
        continue
    match = re.match(r"cf([\d.]+)_margin([\d.]+)", run)
    if not match:
        continue
    cf_val     = float(match.group(1))
    margin_val = float(match.group(2))
    cf_set.add(cf_val)
    margin_set.add(margin_val)

    mean_mauc = get_exp_mean(run_path, metric_key, num_folds)
    mean_or   = read_or_median_from_tde(run_path, num_folds)
    results_mauc[(cf_val, margin_val)] = mean_mauc
    results_or[(cf_val, margin_val)]   = mean_or
    mauc_str = f"{mean_mauc:.2f}%" if mean_mauc is not None else "N/A"
    or_str   = f"{mean_or:.4f}"   if mean_or   is not None else "N/A"
    print(f"  {run}: mAUC={mauc_str}, OR Median={or_str}")

cf_list     = sorted(cf_set)
margin_list = sorted(margin_set)
print(f"\ncf values   : {cf_list}")
print(f"margin values: {margin_list}")

# ---------- 构建矩阵 ----------
R, C = len(cf_list), len(margin_list)
abs_mauc   = np.full((R, C), np.nan)
diff_mauc  = np.full((R, C), np.nan)
abs_or     = np.full((R, C), np.nan)
diff_or    = np.full((R, C), np.nan)

for i, c in enumerate(cf_list):
    for j, m in enumerate(margin_list):
        v_mauc = results_mauc.get((c, m))
        v_or   = results_or.get((c, m))
        if v_mauc is not None:
            abs_mauc[i, j]  = v_mauc
            diff_mauc[i, j] = v_mauc - q2l_mauc
        if v_or is not None:
            abs_or[i, j] = v_or
            if q2l_or_median is not None:
                diff_or[i, j] = v_or - q2l_or_median

# ---------- 绘图工具 ----------
def make_annot(matrix, fmt="{:+.4f}"):
    ann = np.empty(matrix.shape, dtype=object)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            ann[i, j] = fmt.format(v) if not np.isnan(v) else ""
    return ann

xlabels = [f"{m:.2f}" for m in margin_list]
ylabels = [f"{c:.2f}" for c in cf_list]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 子图1：Δ mAUC vs Q2L
vmax_mauc = np.nanmax(np.abs(diff_mauc)) if not np.isnan(diff_mauc).all() else 0.01
sns.heatmap(
    diff_mauc,
    annot=make_annot(diff_mauc, fmt="{:+.2f}"),
    fmt="",
    xticklabels=xlabels, yticklabels=ylabels,
    cmap="RdYlGn", center=0,
    vmin=-vmax_mauc, vmax=vmax_mauc,
    annot_kws={"fontsize": 20},
    ax=axes[0],
)
axes[0].set_title(f"Δ mAUC (%) vs Q2L ({q2l_mauc:.2f}%)", fontsize=20)
axes[0].set_xlabel("margin", fontsize=20)
axes[0].set_ylabel("r", fontsize=20)
axes[0].tick_params(labelsize=16)
axes[0].collections[0].colorbar.ax.tick_params(labelsize=16)
# 子图2：Δ OR Median vs Q2L
if q2l_or_median is not None and not np.isnan(diff_or).all():
    vmax_or = np.nanmax(np.abs(diff_or)) if not np.isnan(diff_or).all() else 0.01
    sns.heatmap(
        diff_or,
        annot=make_annot(diff_or, fmt="{:+.2f}"),
        fmt="",
        xticklabels=xlabels, yticklabels=ylabels,
        cmap="RdYlGn", center=0,
        vmin=-vmax_or, vmax=vmax_or,
        annot_kws={"fontsize": 20},
        ax=axes[1],
    )
    axes[1].set_title(f"Δ OR Median vs Q2L ({q2l_or_median:.2f})", fontsize=20)
else:
    # Q2L 没有 OR Median 时，直接画绝对值
    sns.heatmap(
        abs_or,
        annot=make_annot(abs_or, fmt="{:.4f}"),
        fmt="",
        xticklabels=xlabels, yticklabels=ylabels,
        cmap="YlGnBu",
        annot_kws={"fontsize": 20},
        ax=axes[1],
    )
    axes[1].set_title("OR Median (absolute)", fontsize=20)
axes[1].set_xlabel("margin", fontsize=20)
axes[1].set_ylabel("r", fontsize=20)
axes[1].tick_params(labelsize=18)
axes[1].collections[0].colorbar.ax.tick_params(labelsize=18)
# plt.suptitle(os.path.basename(root_dir), fontsize=17, y=1.02)
plt.tight_layout()
save_path = os.path.join(root_dir, "cf_margin_vs_Q2L.png")
plt.savefig(save_path, bbox_inches="tight", dpi=150)
print(f"\n热力图已保存至: {save_path}")
plt.show()

# ---------- 最优参数 ----------
if not np.isnan(diff_mauc).all():
    max_idx = np.unravel_index(np.nanargmax(diff_mauc), diff_mauc.shape)
    print(f"\n最优组合 (Δ mAUC 最大): cf={cf_list[max_idx[0]]}, margin={margin_list[max_idx[1]]}, "
          f"mAUC={abs_mauc[max_idx]:.2f}%, Δ mAUC={diff_mauc[max_idx]:+.2f}%")
else:
    print("未找到有效数据。")

