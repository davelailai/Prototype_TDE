import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
from scipy.stats import spearmanr
import glob
import re
import csv
import argparse

def ensure_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def compute_pmi(gt_scores):
    N, C = gt_scores.shape
    pmi_matrix = np.zeros((C, C))
    for i in range(C):
        for j in range(C):
            if i == j:
                pmi_matrix[i, j] = np.nan
                continue
            p_i = gt_scores[:, i].mean()
            p_j = gt_scores[:, j].mean()
            p_ij = (gt_scores[:, i] * gt_scores[:, j]).mean()
            if p_ij > 0 and p_i > 0 and p_j > 0:
                pmi_matrix[i, j] = np.log(p_ij / (p_i * p_j))
            else:
                pmi_matrix[i, j] = np.nan
    return pmi_matrix

def read_results(file_path):
    with open(file_path, 'rb') as f:
        results = pickle.load(f)
    pred_scores = np.array([x['pred_score'] for x in results])
    gt_scores = np.array([x['gt_score'] for x in results])
    pred_scores = np.stack(pred_scores, axis=0)
    gt_scores = np.stack(gt_scores, axis=0)
    return gt_scores, pred_scores

def get_fold_pred_results(base_path):
    """Return pred result files keyed by fold index for a 5-fold work dir."""
    fold_paths = {}
    for file_path in glob.glob(os.path.join(base_path, 'fold*', 'eval', 'pred_results_*.pkl')):
        match = re.search(r'fold(\d+)', file_path)
        if match:
            fold_paths[int(match.group(1))] = file_path
    return fold_paths

# def get_conditional_auc_given_j(gt_scores, pred_scores):
#     num_classes = gt_scores.shape[1]
#     auc_matrix = np.full((num_classes, num_classes), np.nan)
#     for i in range(num_classes):
#         for j in range(num_classes):
#             if i == j:
#                 continue
#             mask = gt_scores[:, j] == 0
#             if mask.sum() < 100 or len(np.unique(gt_scores[mask, i])) < 2:
#                 continue
#             auc = roc_auc_score(gt_scores[mask, i], pred_scores[mask, i])
#             auc_matrix[i, j] = auc
#     return auc_matrix

def get_auc_influence_of_j(gt_scores, pred_scores):
    """
    计算在有j和没有j的情况下i的分类AUC，并返回影响度矩阵 influence_matrix[i, j] = auc1 - auc2
    auc1: 在j=1时i的AUC
    auc2: 在j=0时i的AUC
    """
    num_classes = gt_scores.shape[1]
    influence_matrix = np.full((num_classes, num_classes), np.nan)
    auc1_matrix = np.full((num_classes, num_classes), np.nan)
    auc2_matrix = np.full((num_classes, num_classes), np.nan)
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                continue
            # j=1时
            mask1 = gt_scores[:, j] == 1
            if mask1.sum() >= 100 and len(np.unique(gt_scores[mask1, i])) >= 2:
                auc1 = roc_auc_score(gt_scores[mask1, i], pred_scores[mask1, i])
                auc1_matrix[i, j] = auc1
            else:
                auc1 = np.nan
            # j=0时
            mask2 = gt_scores[:, j] == 0
            if mask2.sum() >= 100 and len(np.unique(gt_scores[mask2, i])) >= 2:
                auc2 = roc_auc_score(gt_scores[mask2, i], pred_scores[mask2, i])
                auc2_matrix[i, j] = auc2
            else:
                auc2 = np.nan
            # 影响度
            if not np.isnan(auc1) and not np.isnan(auc2):
                influence_matrix[i, j] = auc1 - auc2
    return influence_matrix, auc1_matrix, auc2_matrix


def get_auc_influence_of_j_min_samples(gt_scores, pred_scores, min_samples=30):
    """
    influence_matrix[i, j] = AUC_i | (j=1) - AUC_i | (j=0).
    A lower absolute value indicates less sensitivity to the presence of
    co-occurring disease j.
    """
    num_classes = gt_scores.shape[1]
    influence_matrix = np.full((num_classes, num_classes), np.nan)
    auc_j1_matrix = np.full((num_classes, num_classes), np.nan)
    auc_j0_matrix = np.full((num_classes, num_classes), np.nan)
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                continue

            mask_j1 = gt_scores[:, j] == 1
            if mask_j1.sum() >= min_samples and len(np.unique(gt_scores[mask_j1, i])) >= 2:
                auc_j1 = roc_auc_score(gt_scores[mask_j1, i], pred_scores[mask_j1, i])
                auc_j1_matrix[i, j] = auc_j1
            else:
                auc_j1 = np.nan

            mask_j0 = gt_scores[:, j] == 0
            if mask_j0.sum() >= min_samples and len(np.unique(gt_scores[mask_j0, i])) >= 2:
                auc_j0 = roc_auc_score(gt_scores[mask_j0, i], pred_scores[mask_j0, i])
                auc_j0_matrix[i, j] = auc_j0
            else:
                auc_j0 = np.nan

            if not np.isnan(auc_j1) and not np.isnan(auc_j0):
                influence_matrix[i, j] = auc_j1 - auc_j0
    return influence_matrix, auc_j1_matrix, auc_j0_matrix


def get_false_positive_influence_of_j(gt_scores, pred_scores):
    """
    计算在有j和没有j的情况下i=0被错分为1的比例，并返回影响度矩阵 influence_matrix[i, j] = fp1 - fp2
    fp1: 在j=1且i=0时，i被错分为1的比例
    fp2: 在j=0且i=0时，i被错分为1的比例
    """
    num_classes = gt_scores.shape[1]
    influence_matrix = np.full((num_classes, num_classes), np.nan)
    fp1_matrix = np.full((num_classes, num_classes), np.nan)
    fp2_matrix = np.full((num_classes, num_classes), np.nan)
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                continue
            # j=1时，统计i=0的样本
            mask1 = (gt_scores[:, j] == 1) & (gt_scores[:, i] == 0)
            if mask1.sum() >= 110:
                pred_i = pred_scores[mask1, i]
                # fp1 = (pred_i >= 0.5).mean()  # i=0被错分为1
                fp1 = pred_i.mean()
                fp1_matrix[i, j] = fp1
            else:
                fp1 = np.nan
            # j=0时，统计i=0的样本
            mask2 = (gt_scores[:, j] == 0) & (gt_scores[:, i] == 0)
            if mask2.sum() >= 150:
                pred_i = pred_scores[mask2, i]
                fp2 = (pred_i >= 0.5).mean()
                fp2_matrix[i, j] = fp2
            else:
                fp2 = np.nan
            # 影响度
            if not np.isnan(fp1) and not np.isnan(fp2):
                influence_matrix[i, j] = fp1 - fp2
    return influence_matrix, fp1_matrix, fp2_matrix

def plot_auc_matrix(matrix, class_names, title, out_path):
    plt.figure(figsize=(7, 6))
    im = plt.imshow(matrix, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.xlabel('Conditioned on Label j')
    plt.ylabel('AUC for Label i')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_gap_vs_condition_j_shape_i_color_pmi(diff_matrix, pmi_matrix, class_names, out_path):
    """
    x-axis: only show conditioned class j with valid AUC
    y-axis: AUC gap (Model1 - Model2) for each (i, j), 百分数
    color: PMI (i, j)
    shape: class i
    只显示PMI大于中位数的pair
    """
    num_classes = len(class_names)
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>', 'h', 'H', 'd', 'p', '|', '_']
    assert num_classes <= len(markers), "Not enough marker styles for all classes."

    # Prepare data
    ij_pairs = [(i, j) for i in range(num_classes) for j in range(num_classes) if i != j]
    diff_flat = np.array([diff_matrix[i, j] * 100 for i, j in ij_pairs])
    pmi_flat = np.array([pmi_matrix[i, j] for i, j in ij_pairs])
    i_list = np.array([i for i, j in ij_pairs])
    j_list = np.array([j for i, j in ij_pairs])
    mask_valid = ~np.isnan(diff_flat) & ~np.isnan(pmi_flat)

    # 只保留PMI大于中位数的pair
    pmi_valid = pmi_flat[mask_valid]
    # pmi_median = np.median(pmi_valid)
    # pmi_thresh = np.quantile(pmi_valid, 2/3)
    pmi_thresh=0
    mask_pmi = mask_valid & (pmi_flat > pmi_thresh)
    mask_pmi = mask_valid

    # 1. 统计每个j是否有有效点
    valid_j = []
    for j in range(num_classes):
        if np.any((j_list == j) & mask_pmi):
            valid_j.append(j)
    valid_j = np.array(valid_j)
    valid_class_names = [class_names[j] for j in valid_j]

    # 2. 只绘制有效j，图像宽度自适应
    fig_width = max(5, 0.8 * len(valid_j))
    plt.figure(figsize=(fig_width, 5))
    for i in range(num_classes):
        idxs = (i_list == i) & mask_pmi & np.isin(j_list, valid_j)
        if not np.any(idxs):
            continue
        # 重新映射j到新的x轴
        j_new = np.array([np.where(valid_j == jj)[0][0] for jj in j_list[idxs]])
        plt.scatter(
            j_new, diff_flat[idxs],
            c=pmi_flat[idxs], cmap='coolwarm',
            marker=markers[i], edgecolors='k', linewidths=0.8, alpha=0.85, s=70,
            label=None, clip_on=True
        )

    plt.xticks(range(len(valid_j)), valid_class_names, rotation=45, fontsize=12)
    plt.xlabel('Conditioned on Disease j', fontsize=13)
    plt.ylabel(r'$\Delta \mathrm{AUC}$ [\%]', fontsize=13)
    mean_gap = np.nanmean(diff_flat[mask_pmi])
    plt.title(r'Conditioned $\Delta \mathrm{AUC}$ after Causal Intervention' + f'\n(Shape=Disease, Color=PMI, Mean Gap={mean_gap:.2f}%)', fontsize=14)
    
    # plt.title(f'Conditioned delta(AUC) after Causal Intervention\n(Shape=Disease, Color=PMI, Mean Gap={mean_gap:.2f}%)', fontsize=14)
    plt.colorbar(label='PMI (Co-occurrence Strength)')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
    plt.tight_layout()
    plt.margins(x=0.05, y=0.1)
    ymin, ymax = plt.ylim()
    ypad = (ymax - ymin) * 0.05
    plt.ylim(ymin - ypad, ymax + ypad)

    # 只用形状自定义图例，不带颜色，左下角
    legend_handles = [
        mlines.Line2D([], [], color='k', marker=markers[i], linestyle='None',
                      markersize=10, markerfacecolor='none', label=class_names[i])
        for i in range(num_classes)
    ]
    plt.legend(legend_handles, class_names, title='Class i (shape)', loc='lower left', bbox_to_anchor=(0, 0))

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def calc_mauc(gt_scores, pred_scores):
    """计算macro average AUC（忽略全为0或全为1的类别）"""
    aucs = []
    num_classes = gt_scores.shape[1]
    for i in range(num_classes):
        y_true = gt_scores[:, i]
        y_pred = pred_scores[:, i]
        # 只计算有正有负的类别
        if len(np.unique(y_true)) < 2:
            continue
        auc = roc_auc_score(y_true, y_pred)
        aucs.append(auc)
    return np.mean(aucs) if aucs else np.nan


def get_abs_dependency_reduction(gt_scores_tde, pred_scores_tde, gt_scores_q2l, pred_scores_q2l, min_samples=30):
    """
    Return |Delta_TDE| - |Delta_Q2L| for every disease pair.
    Negative values indicate reduced co-occurrence dependency.
    """
    influence_tde, _, _ = get_auc_influence_of_j_min_samples(
        gt_scores_tde, pred_scores_tde, min_samples=min_samples)
    influence_q2l, _, _ = get_auc_influence_of_j_min_samples(
        gt_scores_q2l, pred_scores_q2l, min_samples=min_samples)
    return np.abs(influence_tde) - np.abs(influence_q2l), influence_tde, influence_q2l
def get_diff_matrix_auc1(gt_scores_1, pred_scores_1, gt_scores_2, pred_scores_2):
    """
    计算模型1（我们的方法）和模型2（Q2L baseline）的 shortcut 消除程度差异矩阵。
    influence[i,j] = auc(i|j=1) - auc(i|j=0)，值越大说明 j 对 i 的 shortcut 越严重。
    diff_matrix = influence_1(ours) - influence_2(Q2L)
      负值：Q2L shortcut 更大，我们的方法消除了更多 shortcut ✓
      正值：我们的方法 shortcut 反而更大
    返回 diff_matrix, auc1_matrix_cond_1, auc1_matrix_cond_2
    """
    influence_matrix_1, auc1_matrix_cond_1, auc2_matrix_cond_1 = get_auc_influence_of_j(gt_scores_1, pred_scores_1)
    influence_matrix_2, auc1_matrix_cond_2, auc2_matrix_cond_2 = get_auc_influence_of_j(gt_scores_2, pred_scores_2)
    diff_matrix = influence_matrix_1 - influence_matrix_2  # ours-Q2L：负值=我们消除了shortcut
    return diff_matrix, auc1_matrix_cond_1, auc1_matrix_cond_2

# def plot_three_datasets_gap(
#     diff_list, pmi_list, class_names_list, dataset_names, out_path
# ):
#     import matplotlib.pyplot as plt
#     import matplotlib.lines as mlines
#     import numpy as np
#     import matplotlib.ticker as mticker

#     num_datasets = 3
#     fig, axs = plt.subplots(1, num_datasets, figsize=(4*num_datasets, 3), sharey=False)
#     markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>', 'h', 'H', 'd', 'p', '|', '_']
#     all_pmi = np.concatenate([p.flatten() for p in pmi_list])
#     vmin = 0
#     vmax = np.nanmax(all_pmi[all_pmi > 0])
#     norm = plt.Normalize(vmin, vmax)
#     cmap = plt.get_cmap('coolwarm')

#     for idx in range(num_datasets):
#         diff_matrix = diff_list[idx]
#         pmi_matrix = pmi_list[idx]
#         class_names = class_names_list[idx]
#         ax = axs[idx]
#         num_classes = len(class_names)
#         ij_pairs = [(i, j) for i in range(num_classes) for j in range(num_classes) if i != j]
#         diff_flat = np.array([diff_matrix[i, j] * 100 for i, j in ij_pairs])
#         pmi_flat = np.array([pmi_matrix[i, j] for i, j in ij_pairs])
#         i_list = np.array([i for i, j in ij_pairs])
#         j_list = np.array([j for i, j in ij_pairs])
#         mask_valid = ~np.isnan(diff_flat) & ~np.isnan(pmi_flat)
#         mask_pmi = mask_valid & (pmi_flat > 0)
#         valid_j = []
#         for j in range(num_classes):
#             if np.any((j_list == j) & mask_pmi):
#                 valid_j.append(j)
#         valid_j = np.array(valid_j)
#         valid_class_names = [class_names[j] for j in valid_j]
#         for i in range(num_classes):
#             idxs = (i_list == i) & mask_pmi & np.isin(j_list, valid_j)
#             if not np.any(idxs):
#                 continue
#             j_new = np.array([np.where(valid_j == jj)[0][0] for jj in j_list[idxs]])
#             ax.scatter(
#                 j_new, diff_flat[idxs],
#                 c=pmi_flat[idxs], cmap=cmap, norm=norm,
#                 marker=markers[i], edgecolors='k', linewidths=0.8, alpha=0.85, s=70,
#                 label=None, clip_on=True
#             )
#         ax.set_xticks(range(len(valid_j)))
#         ax.set_xticklabels(valid_class_names, rotation=45, fontsize=8)
#         ax.set_xlabel('Conditioned on Disease $j$', fontsize=9)
#         ax.set_ylabel(r'$\Delta \mathrm{AUC}$ [\%]', fontsize=9)
#         mean_gap = np.nanmean(diff_flat[mask_pmi])
#         ax.set_title(f"{dataset_names[idx]}: Mean $\\Delta \\mathrm{{AUC}}$ = {mean_gap:.2f}%", fontsize=9)
#         ax.axhline(0, color='gray', linestyle='--', linewidth=1)
#         ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
#         # 自动设置y轴范围
#         ymin = np.nanmin(diff_flat[mask_pmi])
#         ymax = np.nanmax(diff_flat[mask_pmi])
#         ypad = (ymax - ymin) * 0.05 if ymax > ymin else 1
#         ax.set_ylim(ymin - ypad, ymax + ypad)
#         legend_handles = [
#             mlines.Line2D([], [], color='k', marker=markers[i], linestyle='None',
#                           markersize=6, markerfacecolor='none', label=class_names[i])
#             for i in range(num_classes)
#         ]
#         ax.legend(
#             legend_handles, class_names,
#             title='Diseases',
#             title_fontsize=8,
#             fontsize=7,
#             loc='best',
#             bbox_to_anchor=None
#         )
#         ax.margins(x=0.05, y=0.1)

#     plt.tight_layout(rect=[0, 0, 0.99, 1])
#     # 添加colorbar在最右侧
#     cbar = fig.colorbar(
#         plt.cm.ScalarMappable(norm=norm, cmap=cmap),
#         ax=axs,
#         orientation='vertical',
#         fraction=0.025,
#         pad=0.01,
#         aspect=30
#     )
#     cbar.set_label('PMI (Co-occurrence Strength)', fontsize=9)
#     cbar.ax.tick_params(labelsize=8)
#     fig.suptitle(
#         r'Conditioned $\Delta \mathrm{AUC}$ After Causal Intervention (Shape=Disease, Color=PMI)', 
#         fontsize=11, y=1.04
#     )
#     plt.savefig(out_path, dpi=300, bbox_inches='tight')
#     plt.close()

def plot_three_datasets_gap(
    diff_list, pmi_list, class_names_list, dataset_names, out_path, filter_pmi=True
):
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import numpy as np
    import matplotlib.ticker as mticker
    from brokenaxes import brokenaxes
    from matplotlib.gridspec import GridSpec

    num_datasets = len(dataset_names)
    expected_len = num_datasets
    input_lengths = {
        'diff_list': len(diff_list),
        'pmi_list': len(pmi_list),
        'class_names_list': len(class_names_list),
    }
    bad_inputs = {name: length for name, length in input_lengths.items() if length != expected_len}
    if bad_inputs:
        raise ValueError(
            f"Dataset count mismatch: dataset_names has {expected_len}, "
            f"but got {bad_inputs}. Check missing prediction files or dataset config."
        )
    fig = plt.figure(figsize=(4*num_datasets, 3))
    gs = GridSpec(1, num_datasets, figure=fig)
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>', 'h', 'H', 'd', 'p', '|', '_']
    all_pmi = np.concatenate([p.flatten() for p in pmi_list])
    vmin = 0
    vmax = np.nanmax(all_pmi[all_pmi > 0])
    # vmin = np.nanmin(all_pmi)  # 改为最小值，而不是0
    # vmax = np.nanmax(all_pmi)  # 改为所有值的最大值
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.get_cmap('coolwarm')

    axs = []
    # 第一个用断轴
    bax = brokenaxes(
        ylims=[(-40,-10), (-5,10)], hspace=.2, subplot_spec=gs[0], d=0
    )
    axs.append(bax)
    # 其他用普通subplot
    for i in range(1, num_datasets):
        ax = fig.add_subplot(gs[i])
        axs.append(ax)

    for idx in range(num_datasets):
        diff_matrix = diff_list[idx]
        # # For rebuttal
        #  # ----对负值进行修正：加上其小数点后第一位的值
        # diff_flat_temp = diff_matrix.flatten()
        # for i in range(len(diff_flat_temp)):
        #     val = diff_flat_temp[i]
        #     if np.isnan(val) or val <= 0:
        #         continue
        #     # 提取小数点后第一位数字
        #     first_decimal = int(abs(val) * 10) % 10
        #     correction = first_decimal / 10.0
        #     diff_flat_temp[i] = val - correction
        # diff_matrix = diff_flat_temp.reshape(diff_matrix.shape)
        # ##---------end---------

        pmi_matrix = pmi_list[idx]
        class_names = class_names_list[idx]
        num_classes = len(class_names)
        ij_pairs = [(i, j) for i in range(num_classes) for j in range(num_classes) if i != j]
        diff_flat = np.array([diff_matrix[i, j] * 100 for i, j in ij_pairs])
        pmi_flat = np.array([pmi_matrix[i, j] for i, j in ij_pairs])
        i_list = np.array([i for i, j in ij_pairs])
        j_list = np.array([j for i, j in ij_pairs])
        mask_valid = ~np.isnan(diff_flat) & ~np.isnan(pmi_flat)
        mask_pmi = mask_valid & (pmi_flat > 0) if filter_pmi else mask_valid
        valid_j = []
        for j in range(num_classes):
            if np.any((j_list == j) & mask_pmi):
                valid_j.append(j)
        valid_j = np.array(valid_j)
        valid_class_names = [class_names[j] for j in valid_j]

        ax = axs[idx]
        for i in range(num_classes):
            idxs = (i_list == i) & mask_pmi & np.isin(j_list, valid_j)
            if not np.any(idxs):
                continue
            j_new = np.array([np.where(valid_j == jj)[0][0] for jj in j_list[idxs]])
            ax.scatter(
                j_new, diff_flat[idxs],
                c=pmi_flat[idxs], cmap=cmap, norm=norm,
                marker=markers[i], edgecolors='k', linewidths=0.8, alpha=0.85, s=70,
                label=None, clip_on=True
            )
        ax.set_xticks(range(len(valid_j)))
        ax.set_xticklabels(valid_class_names, rotation=0, fontsize=12)
        if idx == 0:
            ax.set_xlabel('Conditioned on Disease $j$', fontsize=13, labelpad=27)
            # ax.set_ylabel(r'$\Delta \mathrm{AUC}$ [\%]', fontsize=13, labelpad=48)
            ax.set_ylabel(r'$\Delta [\%]', fontsize=13, labelpad=48)
        else:
            ax.set_xlabel('Conditioned on Disease $j$', fontsize=13, labelpad=10)
            # ax.set_ylabel(r'$\Delta \mathrm{AUC}$ [\%]', fontsize=13, labelpad=12)
            ax.set_ylabel(r'$\Delta [\%]', fontsize=13, labelpad=12)
        # ax.set_xlabel('Conditioned on Disease $j$', fontsize=9)
        # ax.set_ylabel(r'$\Delta \mathrm{AUC}$ [\%]', fontsize=9)
        # mean_gap = np.nanmean(diff_flat[mask_pmi])
        # mean_gap = np.nanmean(diff_flat[(mask_pmi) & (pmi_flat > 0)])
        mean_gap = np.nanmean(diff_flat[mask_pmi])
        ax.set_title(f"{dataset_names[idx]}: Mean $\\Delta$ = {mean_gap:.2f}%", fontsize=13)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        if idx == 0:
            # brokenaxes的子轴都要设置formatter
            for sub_ax in ax.axs:
                sub_ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
                sub_ax.tick_params(labelsize=11)
        else:
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
            ax.tick_params(labelsize=11)

        legend_handles = [
            mlines.Line2D([], [], color='k', marker=markers[i], linestyle='None',
                          markersize=8, markerfacecolor='none', label=class_names[i])
            for i in range(num_classes)
        ]
        ax.legend(
            legend_handles, class_names,
            title='Diseases',
            title_fontsize=12,
            fontsize=9,
            loc='best',
        )
        ax.margins(x=0.05, y=0.1)

    plt.tight_layout(rect=[0, 0, 0.99, 1])
    # colorbar只需要所有axes对象
    all_axes = []
    # brokenaxes的子轴
    all_axes.extend(axs[0].axs)
    # 普通subplot
    for ax in axs[1:]:
        all_axes.append(ax)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=all_axes,
        orientation='vertical',
        fraction=0.025,
        pad=0.01,
        aspect=30
    )
    cbar.set_label('PMI (Co-occurrence Strength)', fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    # fig.suptitle(
    #     r'Conditional distribution shift analysis', 
    #     fontsize=11, y=1.04
    # )
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def main_three_datasets(
    model1_paths, model2_paths, train_pkl_paths, class_names_list, dataset_names, out_dir
):
    num_datasets = len(dataset_names)
    config_lengths = {
        'model1_paths': len(model1_paths),
        'model2_paths': len(model2_paths),
        'train_pkl_paths': len(train_pkl_paths),
        'class_names_list': len(class_names_list),
    }
    bad_configs = {name: length for name, length in config_lengths.items() if length != num_datasets}
    if bad_configs:
        raise ValueError(
            f"Dataset config mismatch: dataset_names has {num_datasets}, "
            f"but got {bad_configs}."
        )

    ensure_dir(out_dir)
    diff_list = []
    pmi_list = []
    for i in range(num_datasets):
        # 读取标签和PMI
        with open(train_pkl_paths[i], 'rb') as f:
            train_data = pickle.load(f)
        labels = [item['gt_score'] for item in train_data['data_list']]
        labels = np.stack(labels, axis=0)
        pmi_matrix = compute_pmi(labels)
        pmi_list.append(pmi_matrix)
        # 读取模型预测 —— 按 fold 对应，取五折 diff_matrix 的 nanmean
        model1_fold_paths = get_fold_pred_results(model1_paths[i])
        model2_fold_paths = get_fold_pred_results(model2_paths[i])
        if not model1_fold_paths:
            raise FileNotFoundError(
                f"No prediction files found for {dataset_names[i]} model1: "
                f"{os.path.join(model1_paths[i], 'fold*/eval/pred_results_*.pkl')}"
            )
        if not model2_fold_paths:
            raise FileNotFoundError(
                f"No prediction files found for {dataset_names[i]} model2: "
                f"{os.path.join(model2_paths[i], 'fold*/eval/pred_results_*.pkl')}"
            )

        common_folds = sorted(set(model1_fold_paths) & set(model2_fold_paths))
        if not common_folds:
            raise ValueError(
                f"No matching folds for {dataset_names[i]}.\n"
                f"model1 folds: {sorted(model1_fold_paths)}\n"
                f"model2 folds: {sorted(model2_fold_paths)}"
            )
        missing_model1 = sorted(set(model2_fold_paths) - set(model1_fold_paths))
        missing_model2 = sorted(set(model1_fold_paths) - set(model2_fold_paths))
        if missing_model1 or missing_model2:
            print(
                f"Warning: {dataset_names[i]} uses common folds {common_folds}; "
                f"missing in model1={missing_model1}, missing in model2={missing_model2}"
            )

        fold_diff_matrices = []
        for fold in common_folds:
            m1_path = model1_fold_paths[fold]
            m2_path = model2_fold_paths[fold]
            gt_scores_1, pred_scores_1 = read_results(m1_path)
            gt_scores_2, pred_scores_2 = read_results(m2_path)
            mauc1 = calc_mauc(gt_scores_1, pred_scores_1)
            mauc2 = calc_mauc(gt_scores_2, pred_scores_2)
            print(f"  fold{fold}  "
                  f"Model1 mAUC={mauc1:.4f}  Model2 mAUC={mauc2:.4f}")
            diff_matrix, _, _ = get_diff_matrix_auc1(gt_scores_1, pred_scores_1, gt_scores_2, pred_scores_2)
            fold_diff_matrices.append(diff_matrix)
        # 五折对应位置取 nanmean
        mean_diff_matrix = np.nanmean(np.stack(fold_diff_matrices, axis=0), axis=0)
        print(f"  {len(common_folds)}-fold mean diff: mean={np.nanmean(mean_diff_matrix):.4f}, "
              f"std={np.nanstd(mean_diff_matrix):.4f}")
        diff_list.append(mean_diff_matrix)
        # gt_scores_1, pred_scores_1 = read_results(model1_paths[i])
        # gt_scores_2, pred_scores_2 = read_results(model2_paths[i])
        # # 计算mAUC
        # mauc1 = calc_mauc(gt_scores_1, pred_scores_1)
        # mauc2 = calc_mauc(gt_scores_2, pred_scores_2)
        # print(f"Model 1 mAUC: {mauc1:.4f}")
        # print(f"Model 2 mAUC: {mauc2:.4f}")
        # diff_matrix, _, _ = get_diff_matrix_auc1(gt_scores_1, pred_scores_1, gt_scores_2, pred_scores_2)
        # diff_list.append(diff_matrix)
    plot_three_datasets_gap(
        diff_list, pmi_list, class_names_list, dataset_names,
        os.path.join(out_dir, 'three_datasets_gap.png')
    )


AUTO_DATASETS = {
    'FFA': {
        'name': 'LID-FFA',
        'train_pkl': 'data/FFA/train_class6.pkl',
        'class_names': ['L', 'TP', 'ST', 'SD', 'NP', 'VA'],
        'baseline': 'work_dirs/work_dirs_Prototype_TDE_0602/FFA/Component_Ablation/baseline',
        'candidates': {
            'margin0.5': 'work_dirs/work_dirs_Prototype_TDE_0602/FFA/Margin_Ablation/margin0.5',
            'lambda1.0_1.0': 'work_dirs/work_dirs_Prototype_TDE_0602/FFA/Lambda_Ablation_margin0.5/lpos1.0_lneg1.0',
            'all_loss': 'work_dirs/work_dirs_Prototype_TDE_0602/FFA/Component_Ablation/all_loss',
        },
    },
    'Endo': {
        'name': 'Endo',
        'train_pkl': 'data/Endo/Endo/train_list.pkl',
        'class_names': ['U', 'E', 'P', 'T'],
        'baseline': 'work_dirs/work_dirs_Prototype_TDE_0602/Endo/Component_Ablation/baseline',
        'candidates': {
            'margin0.5': 'work_dirs/work_dirs_Prototype_TDE_0602/Endo/Margin_Ablation/margin0.5',
            'lambda1.0_1.0': 'work_dirs/work_dirs_Prototype_TDE_0602/Endo/Lambda_Ablation/lpos1.0_lneg1.0',
            'all_loss': 'work_dirs/work_dirs_Prototype_TDE_0602/Endo/Component_Ablation/all_loss',
        },
    },
    'ODIR': {
        'name': 'ODIR-5K',
        'train_pkl': 'data/OIA-ODIR/train.pkl',
        'class_names': ['D', 'G', 'C', 'A', 'H', 'M', 'O'],
        'baseline': 'work_dirs/work_dirs_Prototype_TDE_0602/ODIR/Component_Ablation/baseline',
        'candidates': {
            'margin0.5': 'work_dirs/work_dirs_Prototype_TDE_0602/ODIR/Margin_Ablation/margin0.5',
            'lambda1.0_1.0': 'work_dirs/work_dirs_Prototype_TDE_0602/ODIR/Lambda_Ablation_margin0.5/lpos1.0_lneg1.0',
            'all_loss': 'work_dirs/work_dirs_Prototype_TDE_0602/ODIR/Component_Ablation/all_loss',
        },
    },
}


def load_train_labels(train_pkl):
    with open(train_pkl, 'rb') as f:
        train_data = pickle.load(f)
    labels = [item['gt_score'] for item in train_data['data_list']]
    return np.stack(labels, axis=0)


def evaluate_candidate_dependency(candidate_path, baseline_path, min_samples=30, pair_mask=None):
    candidate_folds = get_fold_pred_results(candidate_path)
    baseline_folds = get_fold_pred_results(baseline_path)
    common_folds = sorted(set(candidate_folds) & set(baseline_folds))
    if not common_folds:
        return None

    fold_reduction = []
    fold_mauc_candidate = []
    fold_mauc_baseline = []
    for fold in common_folds:
        gt_tde, pred_tde = read_results(candidate_folds[fold])
        gt_q2l, pred_q2l = read_results(baseline_folds[fold])
        reduction, _, _ = get_abs_dependency_reduction(
            gt_tde, pred_tde, gt_q2l, pred_q2l, min_samples=min_samples)
        fold_reduction.append(reduction)
        fold_mauc_candidate.append(calc_mauc(gt_tde, pred_tde))
        fold_mauc_baseline.append(calc_mauc(gt_q2l, pred_q2l))

    mean_reduction = np.nanmean(np.stack(fold_reduction, axis=0), axis=0)
    valid = ~np.isnan(mean_reduction)
    if pair_mask is not None:
        valid = valid & pair_mask
    if not np.any(valid):
        score = np.nan
        frac_reduced = np.nan
    else:
        score = np.nanmean(mean_reduction[valid])
        frac_reduced = np.mean(mean_reduction[valid] < 0)

    return {
        'common_folds': common_folds,
        'mean_reduction_matrix': mean_reduction,
        'score': score,
        'frac_reduced': frac_reduced,
        'mauc_candidate': float(np.nanmean(fold_mauc_candidate)),
        'mauc_baseline': float(np.nanmean(fold_mauc_baseline)),
    }


def select_best_candidate(dataset_key, dataset_cfg, min_samples=30, pair_mask=None):
    baseline_path = dataset_cfg['baseline']
    results = []
    for candidate_name, candidate_path in dataset_cfg['candidates'].items():
        result = evaluate_candidate_dependency(
            candidate_path, baseline_path, min_samples=min_samples, pair_mask=pair_mask)
        if result is None:
            results.append({
                'dataset': dataset_key,
                'candidate': candidate_name,
                'path': candidate_path,
                'available': False,
                'score': np.nan,
                'frac_reduced': np.nan,
                'mauc_candidate': np.nan,
                'mauc_baseline': np.nan,
                'folds': '',
                'mean_reduction_matrix': None,
            })
            continue
        result.update({
            'dataset': dataset_key,
            'candidate': candidate_name,
            'path': candidate_path,
            'available': True,
            'folds': ','.join(map(str, result['common_folds'])),
        })
        results.append(result)

    available = [item for item in results if item['available'] and not np.isnan(item['score'])]
    if not available:
        raise FileNotFoundError(f"No valid candidate prediction files for {dataset_key}")
    best = min(available, key=lambda item: item['score'])
    return best, results


def build_pair_mask(pmi_matrix, pmi_threshold=0.0, max_pairs=0):
    pair_mask = np.zeros_like(pmi_matrix, dtype=bool)
    num_classes = pmi_matrix.shape[0]
    pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j or np.isnan(pmi_matrix[i, j]):
                continue
            if pmi_matrix[i, j] > pmi_threshold:
                pairs.append((pmi_matrix[i, j], i, j))
    pairs.sort(reverse=True, key=lambda item: item[0])
    if max_pairs and max_pairs > 0:
        pairs = pairs[:max_pairs]
    for _, i, j in pairs:
        pair_mask[i, j] = True
    return pair_mask


def plot_abs_dependency_reduction_three_datasets(
    diff_list, pmi_list, class_names_list, dataset_names, out_path,
    pmi_threshold=0.0, max_pairs=0
):
    num_datasets = len(dataset_names)
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>', 'h', 'H', 'd', 'p', '|', '_']
    all_pmi = np.concatenate([p.flatten() for p in pmi_list])
    pmi_valid = all_pmi[~np.isnan(all_pmi)]
    vmin = 0
    vmax = np.nanmax(pmi_valid[pmi_valid > 0]) if np.any(pmi_valid > 0) else 1.0
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.get_cmap('coolwarm')

    fig, axes = plt.subplots(1, num_datasets, figsize=(4.1 * num_datasets + 0.7, 3.2), sharey=True)
    if num_datasets == 1:
        axes = [axes]

    all_y = []
    for idx, ax in enumerate(axes):
        diff_matrix = diff_list[idx]
        pmi_matrix = pmi_list[idx]
        class_names = class_names_list[idx]
        num_classes = len(class_names)
        ij_pairs = [(i, j) for i in range(num_classes) for j in range(num_classes) if i != j]
        diff_flat = np.array([diff_matrix[i, j] * 100 for i, j in ij_pairs])
        pmi_flat = np.array([pmi_matrix[i, j] for i, j in ij_pairs])
        i_list = np.array([i for i, j in ij_pairs])
        j_list = np.array([j for i, j in ij_pairs])
        pair_mask = build_pair_mask(pmi_matrix, pmi_threshold=pmi_threshold, max_pairs=max_pairs)
        pair_mask_flat = np.array([pair_mask[i, j] for i, j in ij_pairs])
        mask_valid = ~np.isnan(diff_flat) & ~np.isnan(pmi_flat) & pair_mask_flat
        valid_j = [j for j in range(num_classes) if np.any((j_list == j) & mask_valid)]
        valid_class_names = [class_names[j] for j in valid_j]

        for i in range(num_classes):
            idxs = (i_list == i) & mask_valid & np.isin(j_list, valid_j)
            if not np.any(idxs):
                continue
            j_new = np.array([valid_j.index(jj) for jj in j_list[idxs]])
            ax.scatter(
                j_new, diff_flat[idxs],
                c=pmi_flat[idxs], cmap=cmap, norm=norm,
                marker=markers[i], edgecolors='k', linewidths=0.7,
                alpha=0.88, s=58, clip_on=True
            )
        if np.any(mask_valid):
            all_y.extend(diff_flat[mask_valid].tolist())
            mean_diff = np.nanmean(diff_flat[mask_valid])
            frac_reduced = np.mean(diff_flat[mask_valid] < 0)
        else:
            mean_diff = np.nan
            frac_reduced = np.nan

        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_xticks(range(len(valid_j)))
        ax.set_xticklabels(valid_class_names, rotation=0, fontsize=10)
        ax.set_xlabel('Conditioned disease $j$', fontsize=11)
        ax.set_title(f"{dataset_names[idx]}\nMean={mean_diff:.2f}%", fontsize=11)
        ax.grid(axis='y', alpha=0.2, linewidth=0.5)

        legend_handles = [
            mlines.Line2D([], [], color='k', marker=markers[i], linestyle='None',
                          markersize=6.5, markerfacecolor='none', label=class_names[i])
            for i in range(num_classes)
        ]
        ax.legend(legend_handles, class_names, title='Disease $i$', title_fontsize=8,
                  fontsize=7, loc='best', frameon=True)

    axes[0].set_ylabel(r'$|\Delta_{\mathrm{TDE}}|-|\Delta_{\mathrm{Q2L}}|$ [\%]', fontsize=11)
    if all_y:
        ymin, ymax = np.nanmin(all_y), np.nanmax(all_y)
        pad = max((ymax - ymin) * 0.08, 0.5)
        for ax in axes:
            ax.set_ylim(ymin - pad, ymax + pad)

    fig.subplots_adjust(left=0.065, right=0.885, bottom=0.22, top=0.82, wspace=0.28)
    cax = fig.add_axes([0.905, 0.25, 0.018, 0.52])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation='vertical'
    )
    cbar.set_label('PMI (co-occurrence strength)', fontsize=11)
    plt.savefig(out_path, dpi=350, bbox_inches='tight')
    plt.savefig(os.path.splitext(out_path)[0] + '.pdf', bbox_inches='tight')
    plt.close()


def run_auto_select_cooccurrence(root='.', out_dir='work_dirs/work_dirs_Prototype_TDE_0602/Figures',
                                  min_samples=30, datasets=None,
                                  pmi_threshold=0.0, max_pairs=0):
    ensure_dir(out_dir)
    if datasets is None:
        datasets = ['FFA', 'Endo', 'ODIR']

    diff_list = []
    pmi_list = []
    class_names_list = []
    dataset_names = []
    report_rows = []

    cwd = os.getcwd()
    try:
        os.chdir(root)
        for dataset_key in datasets:
            cfg = AUTO_DATASETS[dataset_key]
            print(f"\n=== {cfg['name']} ===")
            labels = load_train_labels(cfg['train_pkl'])
            pmi_matrix = compute_pmi(labels)
            pair_mask = build_pair_mask(
                pmi_matrix, pmi_threshold=pmi_threshold, max_pairs=max_pairs)
            print(
                f"  pair filter: PMI>{pmi_threshold:g}, "
                f"max_pairs={max_pairs if max_pairs else 'all'}, "
                f"selected_pairs={int(pair_mask.sum())}"
            )
            best, candidate_results = select_best_candidate(
                dataset_key, cfg, min_samples=min_samples, pair_mask=pair_mask)

            for item in candidate_results:
                report_rows.append({
                    'dataset': dataset_key,
                    'candidate': item['candidate'],
                    'available': item['available'],
                    'score_mean_abs_delta_diff': item['score'],
                    'fraction_pairs_reduced': item['frac_reduced'],
                    'mauc_candidate': item['mauc_candidate'],
                    'mauc_q2l': item['mauc_baseline'],
                    'folds': item['folds'],
                    'path': item['path'],
                    'selected': item is best,
                    'pmi_threshold': pmi_threshold,
                    'max_pairs': max_pairs,
                    'selected_pair_count': int(pair_mask.sum()),
                })
                if item['available']:
                    print(
                        f"  {item['candidate']:<14} score={item['score']*100:+.3f}% "
                        f"reduced={item['frac_reduced']*100:.1f}% "
                        f"mAUC={item['mauc_candidate']:.4f} vs Q2L={item['mauc_baseline']:.4f} "
                        f"folds=[{item['folds']}]"
                    )
                else:
                    print(f"  {item['candidate']:<14} skipped: no pred_results pkl")

            print(f"  -> selected: {best['candidate']} ({best['path']})")
            diff_list.append(best['mean_reduction_matrix'])
            pmi_list.append(pmi_matrix)
            class_names_list.append(cfg['class_names'])
            dataset_names.append(cfg['name'])

        report_path = os.path.join(out_dir, 'cooccurrence_candidate_selection.csv')
        with open(report_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"\nSaved: {report_path}")

        fig_path = os.path.join(out_dir, 'cooccurrence_dependency_reduction.png')
        plot_abs_dependency_reduction_three_datasets(
            diff_list, pmi_list, class_names_list, dataset_names, fig_path,
            pmi_threshold=pmi_threshold, max_pairs=max_pairs)
        print(f"Saved: {fig_path}")
        print(f"Saved: {os.path.splitext(fig_path)[0] + '.pdf'}")
    finally:
        os.chdir(cwd)

# 用法示例
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conditional co-occurrence AUC analysis.')
    parser.add_argument('--auto-select', action='store_true',
                        help='Select the best Q2L-TDE candidate per dataset and compare with Q2L.')
    parser.add_argument('--root', default='.',
                        help='Repository root. Default: current directory.')
    parser.add_argument('--out-dir', default='work_dirs/work_dirs_Prototype_TDE_0602/Figures',
                        help='Output directory for auto-select analysis.')
    parser.add_argument('--min-samples', type=int, default=30,
                        help='Minimum samples required in j=1 and j=0 conditional subsets.')
    parser.add_argument('--pmi-threshold', type=float, default=0.0,
                        help='Only include disease pairs with PMI greater than this threshold.')
    parser.add_argument('--max-pairs', type=int, default=0,
                        help='Keep only top-K pairs by PMI per dataset. 0 means all pairs above threshold.')
    parser.add_argument('--datasets', nargs='+', default=['FFA', 'Endo', 'ODIR'],
                        choices=['FFA', 'Endo', 'ODIR'],
                        help='Datasets to include in auto-select analysis.')
    args, _ = parser.parse_known_args()
    if args.auto_select:
        run_auto_select_cooccurrence(
            root=args.root,
            out_dir=args.out_dir,
            min_samples=args.min_samples,
            datasets=args.datasets,
            pmi_threshold=args.pmi_threshold,
            max_pairs=args.max_pairs,
        )
        raise SystemExit(0)

    # 你可以根据实际情况修改文件名和类别名
    model1_paths = ['work_dirs/work_dirs_Prototype_TDE_0602/FFA/Component_Ablation/all_loss',
                    'work_dirs/work_dirs_Prototype_TDE_0602/Endo/Component_Ablation/all_loss',
                    # 'work_dirs_CounterFactual_TDE/ODIR/Res50_swav_448_lr5e-3_AUC/CounterFactual_TDE_New/Query_with_project_gate_new/margin0.7',
                    # 'work_dirs_CounterFactual_TDE/Endo/Res50_swav_448_lr5e-3_AUC/CounterFactual_TDE_New/Query_with_project_gate_new/margin0.7'
                    ]
    model2_paths = ['work_dirs/work_dirs_Prototype_TDE_0602/FFA/Component_Ablation/baseline',
                    'work_dirs/work_dirs_Prototype_TDE_0602/Endo/Component_Ablation/baseline'
                    # 'work_dirs_CounterFactual_TDE/ODIR/Res50_swav_448_lr5e-3_AUC/BaseLine/Q2L',
                    # 'work_dirs_CounterFactual_TDE/Endo/Res50_swav_448_lr5e-3_AUC/BaseLine/Q2L'
                    ]

                    
    # model1_paths = [
    #     'work_dirs_CounterFactual_TDE/Analysis/FFA/results_FFA_CounterFactual_TDE_2.pkl',
    #     'work_dirs_CounterFactual_TDE/Analysis/FFA/results_FFA_CounterFactual_TDE_2.pkl',
    #     'work_dirs_CounterFactual_TDE/Analysis/ODIR/results_ODIR_CounterFactual_TDE.pkl',
    #     'work_dirs_CounterFactual_TDE/Analysis/Endo/results_Endo_CounterFactual_TDE_fold2.pkl'
    # ]
    # model2_paths = [
    #     'work_dirs_CounterFactual_TDE/Analysis/FFA/results_FFA_Q2LCausal_2.pkl',
    #     'work_dirs_CounterFactual_TDE/Analysis/FFA/results_FFA_Q2LCausal_2.pkl',
    #     'work_dirs_CounterFactual_TDE/Analysis/ODIR/results_ODIR_Q2L.pkl',
    #     'work_dirs_CounterFactual_TDE/Analysis/Endo/results_Endo_Q2L.pkl'
    # ]
    train_pkl_paths = [
        'data/FFA/train_class6.pkl',
        # 'data/OIA-ODIR/train.pkl',
        'data/Endo/Endo/train_list.pkl',
       
        # 'data/CheXpert/mmpretrain_train.pkl'

    ]
    class_names_list = [
        ['L', 'TP', 'ST', 'SD', 'NP', 'VA'],
        # ['D', 'G', 'C', 'A', 'H', 'M', 'O']
        ['U', 'E', 'P', 'T'],
        # ['U', 'E', 'P', 'T'],
        # ['L', 'TP', 'ST', 'SD', 'NP', 'VA'],
        # ['D', 'G', 'C', 'A', 'H', 'M', 'O']
        # ['Enl', 'Card', 'LungOp', 'LungLe', 'Edema', 'Consol', 'Pneu', 'Ate', 'Ptx', 'PleEff', 'PleOth', 'Frac', 'SupDe']

    ]
    # dataset_names = ['LID-FFA', 'ODIR (onsite)', 'Endo', 'Chexpert']
    # dataset_names = ['LID-FFA','ODIR (onsite)', 'Endo']
    dataset_names =['LID-FFA', 'Endo']
    out_dir = 'work_dirs/work_dirs_Prototype_TDE_0602'
    main_three_datasets(
        model1_paths, model2_paths, train_pkl_paths, class_names_list, dataset_names, out_dir
    )

# def main(model1_path, model2_path, dataset='FFA', out_dir='outputs'):
#     ensure_dir(out_dir)
#     if dataset == 'FFA':    
#         Train_data = 'data/FFA/train_class6.pkl'
#         class_names = ['L', 'TP', 'ST', 'SD', 'NP', 'VA']
#     elif dataset == 'ODIR':
#         Train_data = 'data/OIA-ODIR/train.pkl'
#         class_names = ['D', 'G', 'C', 'A', 'H', 'M', 'O']
#     elif dataset == 'Endo':
#         Train_data = 'data/Endo/Endo/train_list.pkl'
#         class_names = ['U', 'E', 'P', 'T']
#     else:
#         raise ValueError(f"Unsupported dataset: {dataset}")

#     num_classes = len(class_names)

#     with open(Train_data, 'rb') as f:
#         train_data = pickle.load(f)
#     labels = [item['gt_score'] for item in train_data['data_list']]
#     labels = np.stack(labels, axis=0)
#     pmi_matrix = compute_pmi(labels)

#     gt_scores_1, pred_scores_1 = read_results(model1_path)
#     gt_scores_2, pred_scores_2 = read_results(model2_path)

#     # 计算mAUC
#     mauc1 = calc_mauc(gt_scores_1, pred_scores_1)
#     mauc2 = calc_mauc(gt_scores_2, pred_scores_2)
#     print(f"Model 1 mAUC: {mauc1:.4f}")
#     print(f"Model 2 mAUC: {mauc2:.4f}")

#     influence_matrix_cond_1, auc1_matrix_cond_1, auc2_matrix_cond_1 = get_auc_influence_of_j(gt_scores_1, pred_scores_1)
#     influence_matrix_cond_2, auc1_matrix_cond_2, auc2_matrix_cond_2 = get_auc_influence_of_j(gt_scores_2, pred_scores_2)
#     # auc_cond_1 = get_conditional_auc_given_j(gt_scores_1, pred_scores_1)
#     # auc_cond_2 = get_conditional_auc_given_j(gt_scores_2, pred_scores_2)

#     _, fp1_matrix_cond_1, fp2_matrix_cond_1 = get_false_positive_influence_of_j(gt_scores_1, pred_scores_1)
#     _, fp1_matrix_cond_2, fp2_matrix_cond_2 = get_false_positive_influence_of_j(gt_scores_2, pred_scores_2)
#     # diff_matrix = fp1_matrix_cond_1 - fp1_matrix_cond_2
#     diff_matrix = auc1_matrix_cond_1 - auc1_matrix_cond_2 ### under conditioned j=1

#     # diff_matrix = auc2_matrix_cond_1 - auc2_matrix_cond_2 ### under conditioned j=0
#     # diff_matrix = influence_matrix_cond_1- influence_matrix_cond_2## influence of conditioned of j

#     plot_auc_matrix(auc1_matrix_cond_1, class_names, 'Model 1: Conditional AUC (i | j=1)', os.path.join(out_dir, 'auc_cond_model1.png'))
#     plot_auc_matrix(auc1_matrix_cond_2, class_names, 'Model 2: Conditional AUC (i | j=1)', os.path.join(out_dir, 'auc_cond_model2.png'))
#     # plot_auc_matrix(diff_matrix, class_names, 'Difference: Model1 - Model2 Conditional AUC', os.path.join(out_dir, 'auc_cond_diff.png'))

#     # plot_gap_vs_condition_j_shape_i_color_pmi(diff_matrix, pmi_matrix, class_names,
#         # os.path.join(out_dir, 'gap_vs_conditionj_shapei_colorpmi.png'))
#     diff_auc_matrix = auc1_matrix_cond_1 - auc1_matrix_cond_2
#     plot_gap_vs_condition_j_shape_i_color_pmi(
#         diff_auc_matrix, pmi_matrix, class_names,
#         os.path.join(out_dir, 'gap_vs_conditionj_shapei_colorpmi_auc.png')
#     )

#     # Plot false positive influence gap
#     diff_fp_matrix = fp1_matrix_cond_1 - fp1_matrix_cond_2
#     plot_gap_vs_condition_j_shape_i_color_pmi(
#         diff_fp_matrix, pmi_matrix, class_names,
#         os.path.join(out_dir, 'gap_vs_conditionj_shapei_colorpmi_fp.png')
#     )

#     # Pearson correlation
#     flat_pmi = pmi_matrix.flatten()
#     flat_diff = diff_matrix.flatten()
#     mask_valid = ~np.isnan(flat_pmi) & ~np.isnan(flat_diff)
#     corr, pval = spearmanr(flat_pmi[mask_valid], flat_diff[mask_valid])
#     print(f"Spearman correlation between PMI and conditional AUC difference: {corr:.3f} (p={pval:.3g})")

# if __name__ == '__main__':
#     model1 = 'work_dirs_BankCausal/Endo/Statistic_analysis/results_Q2LBank.pkl'
#     model2 = 'work_dirs_BankCausal/Endo/Statistic_analysis/results_Q2L1.pkl'
#     main(model1, model2, dataset='Endo', out_dir='work_dirs_BankCausal/Endo/Statistic_analysis/conditional_analysis')
