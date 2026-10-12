# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, List, Optional, Sequence,Union
from mmengine.optim import OptimWrapper

import torch
from mmengine.device import get_device
from mmengine.dist import get_rank, get_world_size, is_distributed
from mmengine.hooks import Hook
from mmengine.logging import MMLogger

from mmpretrain.registry import HOOKS
from mmpretrain.utils import get_ori_model
from sklearn.cluster import KMeans
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
from collections import defaultdict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

import torch
import numpy as np
from collections import defaultdict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
DATA_BATCH = Optional[Union[dict, tuple, list]]
from PIL import Image

class FeatureTSNEAnalyzer:
    def __init__(self, save_dir, DataName='unKnow'):
        self.save_dir = save_dir
        self.DataName = DataName
        os.makedirs(self.save_dir, exist_ok=True)
        if DataName=='ChestX14':
            self.pathology_list = ['Cardiomegaly','Emphysema','Effusion','Hernia','Nodule',
                    'Pneumothorax','Atelectasis','Pleural_Thickening','Mass','Edema',
                    'Consolidation','Infiltration','Fibrosis','Pneumonia']
            self.index=['A','B','C','D','E','F','G','H','I','J','Q','L','M','N']
        elif DataName=='FFA':
            self.pathology_list = ['Lesion_Leakage','Lesion_Accumulate','Lesion_Dyeing',
                # 'Lesion_Transparent,'
                'Lesion_Shading','Lesion_Inperfusion','Lesion_Vessel_Abnormal']
            self.index=['L','TP','ST','SD','NP','VA']
        elif DataName=='OIA-ODIR':
            self.pathology_list = ['D', 'G', 'C' ,'A', 'H' ,'M', 'O']
            self.index=['D', 'G', 'C' ,'A', 'H' ,'M', 'O']
        elif DataName=='Endo':
            self.pathology_list = ['ulcer','erosion','polyp','tumor']
            self.index=['U','E','P','T']
        else:
            self.pathology_list = []
            self.index=[]

    def aggregate_features(self, features, data_samples):
        all_feats = torch.cat(features, dim=0)
        img_path = [i.img_path for i in data_samples]
        target = torch.stack([i.gt_score.float() for i in data_samples])

        group_feats = defaultdict(list)
        group_targets = defaultdict(list)
        for feat, path, tgt in zip(all_feats, img_path, target):
            group_feats[path].append(feat)
            group_targets[path].append(tgt)

        avg_feats = []
        agg_targets = []
        unique_img_paths = []
        for path in group_feats:
            feats = torch.stack(group_feats[path], dim=0)  # [T, K, D]
            avg_feats.append(feats.mean(dim=0))            # [K, D]
            agg_targets.append(group_targets[path][0])
            unique_img_paths.append(path)
        avg_feats = torch.stack(avg_feats, dim=0).cpu().numpy()      # [num_unique, K, D]
        agg_targets = torch.stack(agg_targets, dim=0).cpu().numpy()  # [num_unique, K]
        return avg_feats, agg_targets, unique_img_paths
    def TDE_statistics(self, TDE_logits, data_sample):
        """
        Args:
            TDE_logits: np.ndarray, shape (B, K, K)
                TDE matrix per sample
            data_sample: dict or tensor
                包含标签信息, shape (B, K)
        Returns:
            class_tde_positive: list, 每个类别的正样本TDE均值
            class_tde_negative: list, 每个类别的负样本TDE均值
            overall_tde_positive: float, 所有对角线上正样本TDE均值
            overall_tde_negative: float, 所有对角线上负样本TDE均值
            overall_off_diag: float, 所有非对角元素的平均TDE
        """

        Avg_TDE, agg_target, _ = self.aggregate_features(TDE_logits, data_sample)
        B, K, _ = Avg_TDE.shape

        # class_tde_positive = []
        # class_tde_negative = []

        # all_pos_tde = []
        # all_neg_tde = []
        # off_diag_vals = []
         # OR 统计
        OR_matrix = np.full((B, K), np.nan)
        direct_effect_matrix = np.full((B, K), np.nan)
        spurious_effects_matrix = np.full((B, K), np.nan)
        for b in range(B):
            labels = agg_target[b]  # [K]
            for k in range(K):
                if labels[k] > 0.5:  # 只计算正样本的OR
                    # 1. 分子: 类别k的直接效应 (TDE₊)
                    direct_effect = Avg_TDE[b, k, k] # 移除p_k对预测k的影响
                    direct_pos = max(direct_effect, 0.0)   # ReLU：避免负OR、避免log1p崩


                    # 2. 分母: 类别k受到的虚假效应的平均强度
                    # 找到所有其他类别的索引 j != k
                    other_classes = np.setdiff1d(np.arange(K), k)
                    # 计算移除每一个其他类别p_j的原型时，对当前类别k预测的影响的绝对值
                    spurious_effects = np.abs(Avg_TDE[b, k, other_classes]) # 这是一个向量
                    # 取这个向量的平均值作为总的虚假效应强度
                    mean_spurious_effect = np.mean(spurious_effects) if len(spurious_effects) > 0 else 1e-6
                    # robust denominator: median (avoid outliers + small mean)
                
                    # or_raw = direct_effect / mean_spurious_effect
                    or_raw = direct_pos /  mean_spurious_effect 
                    or_log = np.log1p(or_raw)  # log(1+OR)，数值稳定、单调

                    # 3. 计算Odds Ratio
                    direct_effect_matrix[b, k] = direct_pos
                    spurious_effects_matrix[b, k] = mean_spurious_effect
                    OR_matrix[b, k] = or_log
        # 计算每一类OR的平均值
        class_or_mean = np.nanmean(OR_matrix, axis=0)  # [K]
        # 计算所有OR的总平均值
        overall_or_mean = np.nanmean(class_or_mean)        # 标量
        # 每一类的中位数、IQR、range
        class_or_median = np.nanmedian(OR_matrix, axis=0)  # [K]
        overall_or_median = np.nanmedian(class_or_median)        # 标量
        # class_or_iqr = np.nanpercentile(OR_matrix, 75, axis=0) - np.nanpercentile(OR_matrix, 25, axis=0)  # [K]
        # class_or_range = np.nanmax(OR_matrix, axis=0) - np.nanmin(OR_matrix, axis=0)  # [K]
        class_tde_possive_mean= np.nanmean(direct_effect_matrix,axis=0)
        overall_TDE_possive_mean= np.nanmean(class_tde_possive_mean)
        class_TDE_negative_mean = np.nanmean(spurious_effects_matrix,axis=0)
        overall_TDE_negative_mean = np.nanmean(class_TDE_negative_mean)

        # # 总体的中位数、IQR、range
        # or_flat = OR_matrix[~np.isnan(OR_matrix)].flatten()
        # overall_or_median = np.median(or_flat)
        # overall_or_iqr = np.percentile(or_flat, 75) - np.percentile(or_flat, 25)
        # overall_or_range = np.max(or_flat) - np.min(or_flat)
        # # 遍历每个类别
        # for class_idx in range(K):
        #     # 对角线 (当前类的TDE)
        #     class_tde_diag = Avg_TDE[:, class_idx, class_idx]   # (B,)
        #     class_labels = agg_target[:, class_idx]             # (B,)

        #     pos_mask = class_labels > 0.5
        #     neg_mask = class_labels <= 0.5

        #     if np.any(pos_mask):
        #         tde_positive = np.mean(class_tde_diag[pos_mask])
        #         all_pos_tde.extend(class_tde_diag[pos_mask].tolist())
        #     else:
        #         tde_positive = 0.0

        #     if np.any(neg_mask):
        #         tde_negative = np.mean(class_tde_diag[neg_mask])
        #         all_neg_tde.extend(class_tde_diag[neg_mask].tolist())
        #     else:
        #         tde_negative = 0.0

        #     class_tde_positive.append(float(tde_positive))
        #     class_tde_negative.append(float(tde_negative))

            

        # # 总体对角线正负
        # overall_tde_positive = float(np.mean(all_pos_tde)) if all_pos_tde else 0.0
        # overall_tde_negative = float(np.mean(all_neg_tde)) if all_neg_tde else 0.0

        # 非对角线整体平均 (类别间干预)
        off_diag_matrix = np.mean(Avg_TDE, axis=0)   # (K, K)
        np.fill_diagonal(off_diag_matrix, 0.0)       # 对角线清零
        return (
            off_diag_matrix,
            class_or_mean,
            overall_or_mean,
            class_or_median,
            overall_or_median,
            class_tde_possive_mean,
            overall_TDE_possive_mean,
            class_TDE_negative_mean,
            overall_TDE_negative_mean
        )
    

    def TDE_statistics_for_spacial(self, TDE_logits_self, TDE_logits_other, data_sample):
        """
        Args:
            TDE_logits: np.ndarray, shape (B, K, K)
                TDE matrix per sample
            data_sample: dict or tensor
                包含标签信息, shape (B, K)
        Returns:
            class_tde_positive: list, 每个类别的正样本TDE均值
            class_tde_negative: list, 每个类别的负样本TDE均值
            overall_tde_positive: float, 所有对角线上正样本TDE均值
            overall_tde_negative: float, 所有对角线上负样本TDE均值
            overall_off_diag: float, 所有非对角元素的平均TDE
        """

        Avg_TDE_self, agg_target, _ = self.aggregate_features(TDE_logits_self, data_sample)
        Avg_TDE_other, agg_target, _ = self.aggregate_features(TDE_logits_other, data_sample)
        B, K= Avg_TDE_self.shape
        
        # class_tde_positive = []
        # class_tde_negative []
        # all_pos_tde = []
        # all_neg_tde = []
        # off_diag_vals = []
         # OR 统计
        OR_matrix = np.full((B, K), np.nan)
        direct_effect_matrix = np.full((B, K), np.nan)
        spurious_effects_matrix = np.full((B, K), np.nan)
        for b in range(B):
            labels = agg_target[b]  # [K]
            for k in range(K):
                if labels[k] > 0.5:  # 只计算正样本的OR
                    # 1. 分子: 类别k的直接效应 (TDE₊)
                    direct_effect = Avg_TDE_self[b, k] # 移除p_k对预测k的影响

                    
                    mean_spurious_effect = np.abs(Avg_TDE_other[b, k]) # 这是一个向量

                    # 3. 计算Odds Ratio
                    direct_effect_matrix[b, k] = direct_effect
                    spurious_effects_matrix[b, k] = mean_spurious_effect
                    OR_matrix[b, k] = direct_effect / (mean_spurious_effect+1e-4)
        # 计算每一类OR的平均值
        class_or_mean = np.nanmean(OR_matrix, axis=0)  # [K]
        # 计算所有OR的总平均值
        overall_or_mean = np.nanmean(class_or_mean)        # 标量
        # 每一类的中位数、IQR、range
        class_or_median = np.nanmedian(OR_matrix, axis=0)  # [K]
        overall_or_median = np.nanmedian(class_or_median)        # 标量
        # class_or_iqr = np.nanpercentile(OR_matrix, 75, axis=0) - np.nanpercentile(OR_matrix, 25, axis=0)  # [K]
        # class_or_range = np.nanmax(OR_matrix, axis=0) - np.nanmin(OR_matrix, axis=0)  # [K]
        class_tde_possive_mean= np.nanmean(direct_effect_matrix,axis=0)
        overall_TDE_possive_mean= np.nanmean(class_tde_possive_mean)
        class_TDE_negative_mean = np.nanmean(spurious_effects_matrix,axis=0)
        overall_TDE_negative_mean = np.nanmean(class_TDE_negative_mean)

        # # 总体的中位数、IQR、range
        # or_flat = OR_matrix[~np.isnan(OR_matrix)].flatten()
        # overall_or_median = np.median(or_flat)
        # overall_or_iqr = np.percentile(or_flat, 75) - np.percentile(or_flat, 25)
        # overall_or_range = np.max(or_flat) - np.min(or_flat)
        # # 遍历每个类别
        # for class_idx in range(K):
        #     # 对角线 (当前类的TDE)
        #     class_tde_diag = Avg_TDE[:, class_idx, class_idx]   # (B,)
        #     class_labels = agg_target[:, class_idx]             # (B,)

        #     pos_mask = class_labels > 0.5
        #     neg_mask = class_labels <= 0.5

        #     if np.any(pos_mask):
        #         tde_positive = np.mean(class_tde_diag[pos_mask])
        #         all_pos_tde.extend(class_tde_diag[pos_mask].tolist())
        #     else:
        #         tde_positive = 0.0

        #     if np.any(neg_mask):
        #         tde_negative = np.mean(class_tde_diag[neg_mask])
        #         all_neg_tde.extend(class_tde_diag[neg_mask].tolist())
        #     else:
        #         tde_negative = 0.0

        #     class_tde_positive.append(float(tde_positive))
        #     class_tde_negative.append(float(tde_negative))

            

        # # 总体对角线正负
        # overall_tde_positive = float(np.mean(all_pos_tde)) if all_pos_tde else 0.0
        # overall_tde_negative = float(np.mean(all_neg_tde)) if all_neg_tde else 0.0

        # 非对角线整体平均 (类别间干预)
        off_diag_matrix =np.nanmean(Avg_TDE_other, axis=0)  # (K, K)
       # 对角线清零
        return (
            off_diag_matrix,
            class_or_mean,
            overall_or_mean,
            class_or_median,
            overall_or_median,
            class_tde_possive_mean,
            overall_TDE_possive_mean,
            class_TDE_negative_mean,
            overall_TDE_negative_mean
        )


    def tsne_and_plot(self, avg_feats, agg_targets, prototypes, queries, epoch):
        import matplotlib.cm as cm
        import math

        pos_feats_all = []
        labels_all = []
        num_classes = queries.shape[0]  # 用queries的shape保证兼容
        # color_map = cm.get_cmap('tab10')
        color_map = cm.get_cmap('tab20', num_classes)         # tab20 支持更多色，第二个参数会采样为 K 个颜色
        # color_map = [mcolors.to_hex(cmap(i)) for i in range(num_classes)]
        if self.index==[]:
            self.index = [str(i) for i in range(num_classes)]
        # 全部类别一起的 t-SNE
        for c in range(num_classes):
            pos_idx = (agg_targets[:, c] > 0.5)
            pos_feats = avg_feats[pos_idx, c, :]  # [N_pos, D]
            pos_feats_all.append(pos_feats)
            labels_all += [f'pos_{self.index[c]}'] * len(pos_feats)
        if len(pos_feats_all) > 0:
            pos_feats_all = np.concatenate(pos_feats_all, axis=0)  # [N_total_pos, D]
        else:
            pos_feats_all = np.zeros((0, queries.shape[1]))
        # 处理prototypes和queries
        if prototypes is not None:
            proto_all = prototypes.cpu().numpy()   # [K, D]
            labels_all += [f'proto_{self.index[i]}' for i in range(num_classes)]
        else:
            proto_all = np.zeros((0, queries.shape[1]))
        query_all = queries.cpu().numpy()      # [K, D]
        labels_all += [f'query_{self.index[i]}' for i in range(num_classes)]
        all_tsne_feats = np.concatenate([pos_feats_all, proto_all, query_all], axis=0)
        tsne = TSNE(n_components=2, random_state=42)
        feats_2d = tsne.fit_transform(all_tsne_feats)
        plt.figure(figsize=(8, 5))   # 扁平宽图，留出右侧 legend 空间
        main_title_size = 18
        subplot_title_size = 12
        label_to_indices = {}
        for idx, label in enumerate(labels_all):
            label_to_indices.setdefault(label, []).append(idx)

        def scatter_label(label, marker, size, alpha=0.5, edgecolors=None, linewidths=None):
            idx = label_to_indices.get(label)
            if not idx:
                return
            cls_token = label.split('_')[1]
            c = self.index.index(cls_token)
            plt.scatter(
                feats_2d[idx, 0],
                feats_2d[idx, 1],
                label=label,
                alpha=alpha,
                s=size,
                marker=marker,
                color=color_map(c),
                edgecolors=edgecolors,
                linewidths=linewidths,
                zorder=10 if marker == '*' else None
            )

        # legend顺序：先queries，再positives（均按疾病顺序）
        for cls_name in self.index:
            scatter_label(f'query_{cls_name}', marker='*', size=200, alpha=1.0, edgecolors='black', linewidths=2)
        for cls_name in self.index:
            scatter_label(f'pos_{cls_name}', marker='o', size=10, alpha=0.5)
        plt.legend(markerscale=2.0, bbox_to_anchor=(1.05, 1), loc='upper left',
                   fontsize=13, title_fontsize=14)
        # plt.title('t-SNE: All Positive Features, Queries', fontsize=main_title_size)
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"tsne_all_pos_proto_query_epoch{epoch}.png")
        plt.savefig(save_path)
        plt.close()

        # 每个类别单独画一个 subplot
        ncols = num_classes
        nrows = math.ceil(num_classes / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
        axes = axes.flatten()
        for c in range(num_classes):
            ax = axes[c]
            pos_idx = (agg_targets[:, c] > 0.5)
            neg_idx = (agg_targets[:, c] <= 0.5)
            pos_feats = avg_feats[pos_idx, c, :]  # [N_pos, D]
            neg_feats = avg_feats[neg_idx, c, :] # [N_neg, D]
            if prototypes is not None:
                proto_c = prototypes[c].cpu().numpy()[None, :]
            else:
                proto_c = np.zeros((0, queries.shape[1]))
            query_c = queries[c].cpu().numpy()[None, :]
            if len(pos_feats) + len(neg_feats) == 0:
                ax.set_title(f'Class {self.index[c]} (no sample)')
                ax.axis('off')
                continue
            feats = np.concatenate([pos_feats, neg_feats, proto_c, query_c], axis=0)
            labels = (
                ['pos_sample'] * len(pos_feats) +
                ['neg_sample'] * len(neg_feats) +
                (['prototype'] if prototypes is not None else []) +
                ['query']
            )
            tsne = TSNE(n_components=2, random_state=42)
            feats_2d = tsne.fit_transform(feats)
            idx_pos = [i for i, l in enumerate(labels) if l == 'pos_sample']
            idx_neg = [i for i, l in enumerate(labels) if l == 'neg_sample']
            idx_proto = [i for i, l in enumerate(labels) if l == 'prototype']
            idx_query = [i for i, l in enumerate(labels) if l == 'query']
            ax.scatter(feats_2d[idx_pos, 0], feats_2d[idx_pos, 1], label='pos_sample', alpha=0.5, s=10, marker='o', color='tab:blue')
            ax.scatter(feats_2d[idx_neg, 0], feats_2d[idx_neg, 1], label='neg_sample', alpha=0.5, s=10, marker='x', color='tab:gray')
            # if prototypes is not None and len(idx_proto) > 0:
            #     ax.scatter(feats_2d[idx_proto, 0], feats_2d[idx_proto, 1], label='prototype', marker='^', s=120, color=color_map(c))
            ax.scatter(feats_2d[idx_query, 0], feats_2d[idx_query, 1], label='query', marker='*', s=120, color=color_map(c))
            ax.set_title(f'Disease {self.index[c]}', fontsize=subplot_title_size)
            ax.legend()
        # 隐藏多余子图
        for i in range(num_classes, len(axes)):
            axes[i].axis('off')
        plt.tight_layout()
        save_path_sub = os.path.join(self.save_dir, f"tsne_per_class_epoch{epoch}.png")
        plt.savefig(save_path_sub)
        plt.close()
        return save_path
    def tsne_intervention_subplot(self, avg_feats, agg_targets, intervened_feats_sub, epoch, save_dir, ncols=4):
        """
        干预前后正负例的 t-SNE 可视化，每个类别一个子图。
        avg_feats: [N, K, D]
        agg_targets: [N, K]
        intervened_feats: [N, K, K, D]
        intervened_feats_sub: [N, K, K, D]
        """
        import math
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from sklearn.manifold import TSNE
        import os

        N, K, D = avg_feats.shape
        ncols = K
        nrows = math.ceil(K / ncols)
        num_classes = K
        title_fontsize = 11
        tick_fontsize = 9
        legend_fontsize = 12
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = axes.flatten()
        if self.index==[]:
            self.index = [str(i) for i in range(num_classes)]
        legend_specs = [
            ('orig_pos', 'tab:blue', 'o', 0.3, 30),
            ('orig_neg', 'tab:blue', 'x', 0.3, 30),
            ('interv_pos', 'tab:orange', 'o', 0.8, 10),
            ('interv_neg', 'tab:orange', 'x', 0.8, 10),
        ]
        legend_handles = [
            Line2D(
                [0], [0],
                marker=marker,
                color='none',
                markerfacecolor=color,
                markeredgecolor='k',
                markeredgewidth=1.0,
                label=label,
                markersize=max(11, size ** 0.5 * 1.8),
                alpha=alpha,
                linestyle='None'
            )
            for label, color, marker, alpha, size in legend_specs
        ]

        for k in range(K):
            pos_mask = agg_targets[:, k] > 0.5
            neg_mask = ~pos_mask

            # 干预前特征
            orig_pos = avg_feats[pos_mask, k, :]  # [N_pos, D]
            orig_neg = avg_feats[neg_mask, k, :]  # [N_neg, D]
            # 干预后特征
            
            interv_pos = intervened_feats_sub[pos_mask, k, k, :]  # 原始正例-原型
            interv_neg = intervened_feats_sub[neg_mask, k, k, :]      # 原始负例-原型


            feats = np.concatenate([orig_pos, orig_neg, interv_pos, interv_neg], axis=0)
            labels = (
                ['orig_pos'] * len(orig_pos) +
                ['orig_neg'] * len(orig_neg) +
                ['interv_pos'] * len(interv_pos) +
                ['interv_neg'] * len(interv_neg)
            )

            if len(feats) < 2:
                axes[k].set_title(f'Class {self.index[k]} (no sample)', fontsize=title_fontsize)
                axes[k].tick_params(axis='both', labelsize=tick_fontsize)
                axes[k].axis('off')
                continue

            tsne = TSNE(n_components=2, random_state=42)
            feats_2d = tsne.fit_transform(feats)
            for label, color, marker, _, _ in legend_specs:
                idx = [i for i, l in enumerate(labels) if l == label]
                if idx:
                    if label.startswith('orig'):
                        axes[k].scatter(
                            feats_2d[idx, 0], feats_2d[idx, 1],
                            c=color, marker=marker, label=label,
                            alpha=0.3, s=30, edgecolors='k', linewidths=0.3
                        )
                    else:  # 干预后
                        axes[k].scatter(
                            feats_2d[idx, 0], feats_2d[idx, 1],
                            c=color, marker=marker, label=label,
                            alpha=0.8, s=10, edgecolors='k', linewidths=0.3
                        )
            axes[k].set_title(f'Disease {self.index[k]}', fontsize=title_fontsize)
            axes[k].tick_params(axis='both', labelsize=tick_fontsize)

        # 隐藏多余子图
        for i in range(K, len(axes)):
            axes[i].axis('off')

        legend = fig.legend(
            handles=legend_handles,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.0),
            ncol=4,
            fontsize=legend_fontsize,
            markerscale=1.2,
            handletextpad=0.6,
            columnspacing=1.2,
            borderpad=0.6,
            frameon=True,
            facecolor='white',
            edgecolor='0.3',
            framealpha=0.95,
            fancybox=False
        )
        legend.get_frame().set_linewidth(1.0)
        plt.tight_layout(rect=(0, 0, 1, 0.9))
        save_path = os.path.join(save_dir, f'tsne_intervention_all_classes_epoch{epoch}.png')
        plt.savefig(save_path)
        plt.close()
    def plot_high_pmi_pairs(self, avg_feats, agg_targets, epoch, topk=5, prototypes=None, queries=None):
        """可视化PMI最高的疾病对的正样本分布（subplot形式），并显示对应的prototype和query"""
        import numpy as np
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import math

        N, K = agg_targets.shape
        color_map = cm.get_cmap('tab10')
        num_classes = K
        if self.index==[]:
            self.index = [str(i) for i in range(num_classes)]
        # 计算PMI
        label_count = agg_targets.sum(axis=0) # [K]
        co_count = (np.expand_dims(agg_targets, 2) * np.expand_dims(agg_targets, 1)).sum(axis=0)
        p_i = label_count / N
        p_j = label_count / N
        p_ij = co_count / N
        eps = 1e-10
        PMI = np.log((p_ij + eps) / (p_i[None, :] * p_j[:, None] + eps))
        np.fill_diagonal(PMI, 0)
        # 取PMI最高的topk对
        triu_indices = np.triu_indices(K, k=1)
        top_indices = np.argsort(PMI[triu_indices])[::-1][:topk]
        co_pairs = list(zip(triu_indices[0][top_indices], triu_indices[1][top_indices]))

        ncols = len(co_pairs)
        nrows = math.ceil(len(co_pairs) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 6*nrows))
        axes = axes.flatten()

        for i, (a, b) in enumerate(co_pairs):
            ax = axes[i]
            idx_a = (agg_targets[:, a] > 0.5)
            idx_b = (agg_targets[:, b] > 0.5)
            feats_a = avg_feats[idx_a, a, :]
            feats_b = avg_feats[idx_b, b, :]
            # 加入prototype和query
            proto_a = prototypes[a].cpu().numpy()[None, :] if prototypes is not None else np.zeros((0, feats_a.shape[1]))
            proto_b = prototypes[b].cpu().numpy()[None, :] if prototypes is not None else np.zeros((0, feats_b.shape[1]))
            query_a = queries[a].cpu().numpy()[None, :] if queries is not None else np.zeros((0, feats_a.shape[1]))
            query_b = queries[b].cpu().numpy()[None, :] if queries is not None else np.zeros((0, feats_b.shape[1]))
            feats = np.concatenate([feats_a, feats_b, proto_a, proto_b, query_a, query_b], axis=0)
            labels = (
                ['disease_%d' % a] * len(feats_a) +
                ['disease_%d' % b] * len(feats_b) +
                ['proto_a'] * len(proto_a) + ['proto_b'] * len(proto_b) +
                ['query_a'] * len(query_a) + ['query_b'] * len(query_b)
            )
            if len(feats) < 2:
                ax.set_title(f'Disease {self.index[a]} vs {self.index[b]} (no sample)')
                ax.axis('off')
                continue
            tsne = TSNE(n_components=2, random_state=42)
            feats_2d = tsne.fit_transform(feats)
            # 正样本
            idx_a_list = [j for j, l in enumerate(labels) if l == f'disease_{a}']
            idx_b_list = [j for j, l in enumerate(labels) if l == f'disease_{b}']
            ax.scatter(feats_2d[idx_a_list, 0], feats_2d[idx_a_list, 1], label=f'disease_{a}', alpha=0.5, s=10, marker='o', color=color_map(a))
            ax.scatter(feats_2d[idx_b_list, 0], feats_2d[idx_b_list, 1], label=f'disease_{b}', alpha=0.5, s=10, marker='o', color=color_map(b))
            # prototype/query 高亮
            idx_proto_a = [j for j, l in enumerate(labels) if l == 'proto_a']
            idx_proto_b = [j for j, l in enumerate(labels) if l == 'proto_b']
            idx_query_a = [j for j, l in enumerate(labels) if l == 'query_a']
            idx_query_b = [j for j, l in enumerate(labels) if l == 'query_b']
            # if idx_proto_a:
            #     ax.scatter(feats_2d[idx_proto_a, 0], feats_2d[idx_proto_a, 1], label='proto_a', marker='^', s=200, color=color_map(a), edgecolors='black', linewidths=2, zorder=10)
            # if idx_proto_b:
            #     ax.scatter(feats_2d[idx_proto_b, 0], feats_2d[idx_proto_b, 1], label='proto_b', marker='^', s=200, color=color_map(b), edgecolors='black', linewidths=2, zorder=10)
            if idx_query_a:
                ax.scatter(feats_2d[idx_query_a, 0], feats_2d[idx_query_a, 1], label='query_a', marker='*', s=200, color=color_map(a), linewidths=2, zorder=10)
            if idx_query_b:
                ax.scatter(feats_2d[idx_query_b, 0], feats_2d[idx_query_b, 1], label='query_b', marker='*', s=200, color=color_map(b), linewidths=2, zorder=10)
            ax.legend()
            ax.set_title(f"t-SNE: Disease {self.index[a]} vs {self.index[b]} (PMI={PMI[a, b]:.2f})")
        # 隐藏多余子图
        for i in range(len(co_pairs), len(axes)):
            axes[i].axis('off')
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"tsne_pmi_pairs_epoch{epoch}.png")
        plt.savefig(save_path)
        plt.close()
    
    def plot_off_diag_matrix(self, off_diag_matrix, class_names=None, title="TDE Off-diagonal Confusion Matrix"):
        """
        可视化非对角 TDE 矩阵为 confusion matrix 风格

        Args:
            off_diag_matrix: np.ndarray, (K, K)，对角线已置零
            class_names: list[str], 类别名称列表（可选）
            title: str, 图标题
        """
        K = off_diag_matrix.shape[0]
        if self.index==[]:
            self.index = [str(i) for i in range(K)]

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(off_diag_matrix, cmap="coolwarm", interpolation="nearest")

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Mean TDE (off-diagonal)")

        # 设置坐标轴
        if class_names is None:
            class_names = [f"{self.index[i]}" for i in range(K)]

        ax.set_xticks(np.arange(K))
        ax.set_yticks(np.arange(K))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)

        ax.set_xlabel("Predicted / Influence from Class j")
        ax.set_ylabel("True Class i")
        ax.set_title(title)

        # 在格子里写数值
        for i in range(K):
            for j in range(K):
                if i != j:  # 只标注非对角
                    ax.text(j, i, f"{off_diag_matrix[i, j]:.2f}",
                            ha="center", va="center", color="black")

        plt.tight_layout()
        # plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"TDE Off-diagonal Confusion Matrix.png")
        plt.savefig(save_path)
        plt.show()

    def forward(self, features, data_samples, prototypes, queries, epoch, plot_pmi=False, pmi_topk=5):
        avg_feats, agg_targets, unique_img_paths = self.aggregate_features(features, data_samples)
        tsne_path = self.tsne_and_plot(avg_feats, agg_targets, prototypes, queries, epoch)
        tsne_path = None
        # if plot_pmi:
        #     self.plot_high_pmi_pairs(avg_feats, agg_targets, epoch, topk=pmi_topk, prototypes=prototypes, queries=queries)
        return avg_feats, agg_targets, unique_img_paths, tsne_path

def Gan_train_step(self, data: Union[dict, tuple, list],                   optim_wrapper: OptimWrapper, **kwargs) -> Dict[str, torch.Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        During non-distributed training. If subclasses do not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        1. Calls ``self.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # Enable automatic mixed precision training context.
        # 
        # with optim_wrapper.optim_context(self):
        #     data = self.data_preprocessor(data, True)
        #     losses = self._run_forward(data, mode='loss')  # type: ignore
        # parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        # optim_wrapper.update_params(parsed_losses)
        # return log_vars
    
        
        #Train causal matrix
        # if self.warmup_epoch:

        if self.current_epoch < self.max_epochs:
            base_optimizer_wrapper = optim_wrapper['backbone']
            with base_optimizer_wrapper.optim_context(self):
                data = self.data_preprocessor(data, True)
                losses = self._run_forward(data, mode='loss')  # type: ignore
                parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
                base_optimizer_wrapper.update_params(parsed_losses)
            return log_vars
        elif self.current_epoch % 2 == 0:
            base_optimizer_wrapper = optim_wrapper['backbone']
            self.set_requires_grad(self.head.DagmaMLP, False)
            with base_optimizer_wrapper.optim_context(self):
                data = self.data_preprocessor(data, True)
                losses = self._run_forward(data, mode='loss')  # type: ignore
                losses.pop('causal_loss')
                parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
                base_optimizer_wrapper.update_params(parsed_losses)
            self.set_requires_grad(self.head.DagmaMLP, True)
            return log_vars
        else:
            base_optimizer_wrapper = optim_wrapper['head']
            # self.head.task_heads.classification.feature_reg=True
            self.set_requires_grad(self.head, False)
            self.set_requires_grad(self.backbone, False)
            self.set_requires_grad(self.head.DagmaMLP, True)

            with base_optimizer_wrapper.optim_context(self):
                data = self.data_preprocessor(data, True)
                losses = self._run_forward(data, mode='loss')  # type: ignore
                # losses.pop('classification_loss')
                parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
                base_optimizer_wrapper.update_params(parsed_losses)
            self.set_requires_grad(self.head, True)
            self.set_requires_grad(self.backbone, True)
            # self.set_requires_grad(self.head.DagmaMLP, True)
            return log_vars

@HOOKS.register_module()
class EpochWarmupHook(Hook):
    """Pass current_epoch and warmup_epoch to model.head for epoch-based training."""

    def __init__(self, warmup_epoch=10,transition_length=5, DataName='UnKnown'):
        self.warmup_epoch = warmup_epoch
        self.transition_length = transition_length
        self.DataName = DataName
        

    def before_train_epoch(self, runner):
        current_epoch = runner.epoch
        model_head = get_ori_model(runner.model).head
        model_head.task_heads['classification'].current_epoch = current_epoch
        model_head.task_heads['classification'].warmup_epoch = self.warmup_epoch
        model_head.task_heads['classification'].transition_length = self.transition_length

        # model_head.current_epoch = current_epoch
        # model_head.warmup_epoch = self.warmup_epoch
    def before_test_epoch(self, runner):
        current_epoch = runner.epoch
        model = get_ori_model(runner.model)
        if hasattr(model, 'head'):
            model_head = model.head
            model_head.task_heads['classification'].current_epoch = current_epoch+100
            model_head.task_heads['classification'].warmup_epoch = self.warmup_epoch
            model_head.task_heads['classification'].transition_length = self.transition_length
        else:
            model_head = model.module.head
            model_head.task_heads['classification'].current_epoch = current_epoch +100
            model_head.task_heads['classification'].warmup_epoch = self.warmup_epoch
            model_head.task_heads['classification'].transition_length = self.transition_length
            # model_head.current_epoch = current_epoch
            # model_head.warmup_epoch = self.warmup_epoch + 100
    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each validation epoch.

        Args:
            runner (Runner): The runner of the validation process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on validation dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        # if runner.epoch % 10 == 0:
        model = get_ori_model(runner.model)
        if hasattr(model, 'head'):
            model_head = model.head
        else:
            model_head = model.module.head
        features = getattr(model_head.task_heads['classification'], 'test_features', None)
        TDE_logits = getattr(model_head.task_heads['classification'], 'test_TDE_logits', None)
        data_samples = getattr(model_head.task_heads['classification'], 'test_data_samples', None)
        analyzer = FeatureTSNEAnalyzer(runner.work_dir,self.DataName)
        off_diag_matrix, class_or_mean, overall_or_mean, class_or_median, overall_or_median, class_tde_possive_mean,overall_TDE_possive_mean,class_TDE_negative_mean, overall_TDE_negative_mean = analyzer.TDE_statistics(TDE_logits, data_samples)
        runner.logger.info(
            f"Epoch {runner.epoch} class_tde_positive - "f"{class_tde_possive_mean},class_tde_negative - "f"{class_TDE_negative_mean}, overall_tde_positive - "f"{overall_TDE_possive_mean}, overall_tde_negative - "f"{overall_TDE_negative_mean}, class_or_mean - "f"{class_or_mean}, overall_or_mean - "f"{overall_or_mean}, class_or_median - "f"{class_or_median},overall_or_median - "f"{overall_or_median}")
        # analyzer.plot_off_diag_matrix(off_diag_matrix)
        # if features is not None and len(features) > 0:
        #     analyzer = FeatureTSNEAnalyzer(runner.work_dir)
        #     prototypes = model_head.task_heads['classification'].get_prototypes().detach().cpu()
        #     queries = model_head.task_heads['classification'].query_embed.weight.detach().cpu()
        #     avg_feats, agg_targets, unique_img_paths, tsne_path = analyzer.forward(
        #         features, data_samples, prototypes, queries, runner.epoch,plot_pmi=True
        #     )
        #     # avg_feats, agg_targets, unique_img_paths = analyzer.aggregate_features(features, data_samples)
        #     B,K,D= avg_feats.shape
        #     proto = model_head.task_heads['classification'].get_prototypes().view(1, 1, K, D).expand(B, K, K, D)
        #     avg_feats_tensor = torch.from_numpy(avg_feats).to(proto.device)  
        #     f_c = avg_feats_tensor.unsqueeze(2).expand(B, K, K, D)  # [B, K, K, D]
        #     intervened_feats = model_head.task_heads['classification'].intervention_combiner(f_c, proto).detach().cpu().numpy() # [B, K, K, D]
        #     intervened_feats_sub = model_head.task_heads['classification'].intervention_combiner(f_c, proto, direction="sub").detach().cpu().numpy()
        #     analyzer.tsne_intervention_subplot(avg_feats, agg_targets, intervened_feats, intervened_feats_sub, runner.epoch, runner.work_dir)
            
        self._after_epoch(runner, mode='val')
    def after_test_epoch(self,
                         runner,
                         metrics: Optional[Dict[str, float]] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each test epoch.

        Args:
            runner (Runner): The runner of the testing process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on test dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        model = get_ori_model(runner.model)
        if hasattr(model, 'head'):
            model_head = model.head
        else:
            model_head = model.module.head
        features = getattr(model_head.task_heads['classification'], 'test_features', None)
       
        
        data_samples = getattr(model_head.task_heads['classification'], 'test_data_samples', None)
        if features is not None and len(features) > 0:
            analyzer = FeatureTSNEAnalyzer(runner.work_dir,self.DataName)
            if hasattr(model_head.task_heads['classification'], 'get_prototypes'):
                prototypes = model_head.task_heads['classification'].get_prototypes().detach().cpu()
            else:
                 prototypes = None
            queries = model_head.task_heads['classification'].query_embed.weight.detach().cpu()
            avg_feats, agg_targets, unique_img_paths, tsne_path = analyzer.forward(
                features, data_samples, prototypes, queries, runner.epoch,plot_pmi=True
            )
            B,K,D= avg_feats.shape
            if prototypes is not None:
            # f_c = avg_feats.unsqueeze(2).expand(B, K, K, D)  # [B, K, K, D]
                proto = model_head.task_heads['classification'].get_prototypes().view(1, 1, K, D).expand(B, K, K, D)
                avg_feats_tensor = torch.from_numpy(avg_feats).to(proto.device)  
                f_c = avg_feats_tensor.unsqueeze(2).expand(B, K, K, D)  # [B, K, K, D]
                # intervened_feats = model_head.task_heads['classification'].intervention_combiner(f_c, proto).detach().cpu().numpy() # [B, K, K, D]
                # 分批计算并搬到 CPU，避免一次性 OOM
                batch_cpu = 2000  # 根据显存调小或调大
                parts = []
                with torch.no_grad():
                    for i in range(0, B, batch_cpu):
                        i2 = min(B, i + batch_cpu)
                        f_chunk = f_c[i:i2].to(proto.device)            # [bs, K, K, D]
                        proto_chunk = proto[i:i2].to(proto.device)
                        out = model_head.task_heads['classification'].intervention_combiner(
                            f_chunk, proto_chunk, direction="sub"
                        )  # [bs, K, K, D]
                        parts.append(out.detach().cpu().numpy())
                        del out, f_chunk, proto_chunk
                        torch.cuda.empty_cache()
                intervened_feats_sub = np.concatenate(parts, axis=0)  # [B, K, K, D]
                # intervened_feats_sub = model_head.task_heads['classification'].intervention_combiner(f_c, proto, direction="sub").detach().cpu().numpy()
                analyzer.tsne_intervention_subplot(avg_feats, agg_targets, intervened_feats_sub, runner.epoch, runner.work_dir)
            
            TDE_logits = getattr(model_head.task_heads['classification'], 'test_TDE_logits', None)
            if TDE_logits is not None:
                off_diag_matrix, class_or_mean, overall_or_mean, class_or_median, overall_or_median, class_tde_possive_mean,overall_TDE_possive_mean,class_TDE_negative_mean, overall_TDE_negative_mean = analyzer.TDE_statistics(TDE_logits, data_samples)
                with open(f"{runner.work_dir}/tde_stats_epoch_{runner.epoch}.md", "w") as f:
                    f.write(
                        f"""### Epoch {runner.epoch} TDE Statistics
                    - **Class TDE Positive:** {class_tde_possive_mean}
                    - **Class TDE Negative:** {class_TDE_negative_mean}
                    - **Overall TDE Positive:** {overall_TDE_possive_mean}
                    - **Overall TDE Negative:** {overall_TDE_negative_mean}
                    - **Class OR Mean:** {class_or_mean}
                    - **Overall OR Mean:** {overall_or_mean}
                    - **Class OR Median:** {class_or_median}
                    - **Overall OR Median:** {overall_or_median}
                    - **TDE Off-diagonal Matrix:** {off_diag_matrix}
                    """
                        )
                # runner.logger.info(
                #     f"Epoch {runner.epoch} class_tde_positive - "f"{class_tde_positive},class_tde_negative - "f"{class_tde_negative}, overall_tde_positive - "f"{overall_tde_positive}, overall_tde_negative - "f"{overall_tde_negative}" )
                # analyzer.plot_off_diag_matrix(off_diag_matrix)

        self._after_epoch(runner, mode='test')

@HOOKS.register_module()
class EpochWarmupHook_Causal_Patch(Hook):
    """Pass current_epoch and warmup_epoch to model.head for epoch-based training."""

    def __init__(self, warmup_epoch=10,transition_length=5, DataName='UnKnown'):
        self.warmup_epoch = warmup_epoch
        self.transition_length = transition_length
        self.DataName = DataName
        

    def before_train_epoch(self, runner):
        current_epoch = runner.epoch
        model_head = get_ori_model(runner.model).head
        model_head.task_heads['classification'].current_epoch = current_epoch
        model_head.task_heads['classification'].warmup_epoch = self.warmup_epoch
        model_head.task_heads['classification'].transition_length = self.transition_length

        # model_head.current_epoch = current_epoch
        # model_head.warmup_epoch = self.warmup_epoch
    def before_test_epoch(self, runner):
        current_epoch = runner.epoch
        model = get_ori_model(runner.model)
        if hasattr(model, 'head'):
            model_head = model.head
            model_head.task_heads['classification'].current_epoch = current_epoch+100
            model_head.task_heads['classification'].warmup_epoch = self.warmup_epoch
            model_head.task_heads['classification'].transition_length = self.transition_length
        else:
            model_head = model.module.head
            model_head.task_heads['classification'].current_epoch = current_epoch +100
            model_head.task_heads['classification'].warmup_epoch = self.warmup_epoch
            model_head.task_heads['classification'].transition_length = self.transition_length
            # model_head.current_epoch = current_epoch
            # model_head.warmup_epoch = self.warmup_epoch + 100
    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each validation epoch.

        Args:
            runner (Runner): The runner of the validation process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on validation dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        # if runner.epoch % 10 == 0:
        model = get_ori_model(runner.model)
        if hasattr(model, 'head'):
            model_head = model.head
        else:
            model_head = model.module.head
        
        TDE_logits_self = getattr(model_head.task_heads['classification'], 'test_TDE_self', None)
        TDE_logits_others = getattr(model_head.task_heads['classification'], 'test_TDE_others', None)
        data_samples = getattr(model_head.task_heads['classification'], 'test_data_samples', None)
        analyzer = FeatureTSNEAnalyzer(runner.work_dir,self.DataName)
        off_diag_matrix, class_or_mean, overall_or_mean, class_or_median, overall_or_median, class_tde_possive_mean,overall_TDE_possive_mean,class_TDE_negative_mean, overall_TDE_negative_mean = analyzer.TDE_statistics_for_spacial(TDE_logits_self, TDE_logits_others,data_samples)
        runner.logger.info(
            f"Epoch {runner.epoch} class_tde_positive - "f"{class_tde_possive_mean},class_tde_negative - "f"{class_TDE_negative_mean}, overall_tde_positive - "f"{overall_TDE_possive_mean}, overall_tde_negative - "f"{overall_TDE_negative_mean}, class_or_mean - "f"{class_or_mean}, overall_or_mean - "f"{overall_or_mean}, class_or_median - "f"{class_or_median},overall_or_median - "f"{overall_or_median}")
        # analyzer.plot_off_diag_matrix(off_diag_matrix)
        # features = getattr(model_head.task_heads['classification'], 'masked_features', None)
        # if features is not None and len(features) > 0:
        #     analyzer = FeatureTSNEAnalyzer(runner.work_dir)
        #     queries = model_head.task_heads['classification'].query_embed.weight.detach().cpu()
        #     avg_feats, agg_targets, unique_img_paths, tsne_path = analyzer.forward(
        #         features, data_samples, None, queries, runner.epoch,plot_pmi=True
        #     )
        #     # avg_feats, agg_targets, unique_img_paths = analyzer.aggregate_features(features, data_samples)
        #     # B,K,D= avg_feats.shape
        #     # proto = model_head.task_heads['classification'].get_prototypes().view(1, 1, K, D).expand(B, K, K, D)
        #     # avg_feats_tensor = torch.from_numpy(avg_feats).to(proto.device)  
        #     # f_c = avg_feats_tensor.unsqueeze(2).expand(B, K, K, D)  # [B, K, K, D]
        #     # intervened_feats = model_head.task_heads['classification'].intervention_combiner(f_c, proto).detach().cpu().numpy() # [B, K, K, D]
        #     # intervened_feats_sub = model_head.task_heads['classification'].intervention_combiner(f_c, proto, direction="sub").detach().cpu().numpy()
        #     intervened_feats_sub= getattr(model_head.task_heads['classification'], 'cf_self_features', None)
        #     # 假设 intervened_feats_sub shape 为 [B, K, D]
        #     intervened_feats_sub, _, _ = analyzer.aggregate_features(intervened_feats_sub, data_samples)
        #     intervened_feats_sub_expanded = np.expand_dims(intervened_feats_sub, axis=2)
        #     intervened_feats_sub_expanded = np.repeat(intervened_feats_sub_expanded, intervened_feats_sub.shape[1], axis=2)
        #     # 现在 shape 为 [B, K, K, D]
        #     analyzer.tsne_intervention_subplot(avg_feats, agg_targets, intervened_feats_sub_expanded, runner.epoch, runner.work_dir)
            
        self._after_epoch(runner, mode='val')
    def after_test_epoch(self,
                         runner,
                         metrics: Optional[Dict[str, float]] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each test epoch.

        Args:
            runner (Runner): The runner of the testing process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on test dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        model = get_ori_model(runner.model)
        if hasattr(model, 'head'):
            model_head = model.head
        else:
            model_head = model.module.head
        features = getattr(model_head.task_heads['classification'], 'masked_features', None)
        
        data_samples = getattr(model_head.task_heads['classification'], 'test_data_samples', None)
        if features is not None and len(features) > 0:
            analyzer = FeatureTSNEAnalyzer(runner.work_dir,self.DataName)
            if hasattr(model_head.task_heads['classification'], 'get_prototypes'):
                prototypes = model_head.task_heads['classification'].get_prototypes().detach().cpu()
            else:
                 prototypes = None
            queries = model_head.task_heads['classification'].query_embed.weight.detach().cpu()
            avg_feats, agg_targets, unique_img_paths, tsne_path = analyzer.forward(
                features, data_samples, prototypes, queries, runner.epoch,plot_pmi=True
            )
            B,K,D= avg_feats.shape

            intervened_feats_sub= getattr(model_head.task_heads['classification'], 'cf_self_features', None)
            intervened_feats_sub, _, _ = analyzer.aggregate_features(intervened_feats_sub, data_samples)
            intervened_feats_sub_expanded = np.expand_dims(intervened_feats_sub, axis=2)
            intervened_feats_sub_expanded = np.repeat(intervened_feats_sub_expanded, intervened_feats_sub.shape[1], axis=2)
            # 现在 shape 为 [B, K, K, D]
            analyzer.tsne_intervention_subplot(avg_feats, agg_targets, intervened_feats_sub_expanded, runner.epoch, runner.work_dir)  
            
            TDE_logits_self = getattr(model_head.task_heads['classification'], 'test_TDE_self', None)
            TDE_logits_others = getattr(model_head.task_heads['classification'], 'test_TDE_others', None)
            if TDE_logits_self is not None and TDE_logits_others is not None:
                off_diag_matrix, class_or_mean, overall_or_mean, class_or_median, overall_or_median, class_tde_possive_mean,overall_TDE_possive_mean,class_TDE_negative_mean, overall_TDE_negative_mean = analyzer.TDE_statistics_for_spacial(TDE_logits_self, TDE_logits_others,data_samples)
                with open(f"{runner.work_dir}/tde_stats_epoch_{runner.epoch}.md", "w") as f:
                    f.write(
                        f"""### Epoch {runner.epoch} TDE Statistics
                    - **Class TDE Positive:** {class_tde_possive_mean}
                    - **Class TDE Negative:** {class_TDE_negative_mean}
                    - **Overall TDE Positive:** {overall_TDE_possive_mean}
                    - **Overall TDE Negative:** {overall_TDE_negative_mean}
                    - **Class OR Mean:** {class_or_mean}
                    - **Overall OR Mean:** {overall_or_mean}
                    - **Class OR Median:** {class_or_median}
                    - **Overall OR Median:** {overall_or_median}
                    - **TDE Off-diagonal Matrix:** {off_diag_matrix}
                    """
                        )
                # runner.logger.info(
                #     f"Epoch {runner.epoch} class_tde_positive - "f"{class_tde_positive},class_tde_negative - "f"{class_tde_negative}, overall_tde_positive - "f"{overall_tde_positive}, overall_tde_negative - "f"{overall_tde_negative}" )
                # analyzer.plot_off_diag_matrix(off_diag_matrix)

        self._after_epoch(runner, mode='test')
    # def after_test_iter(self,
    #                     runner,
    #                     batch_idx: int,
    #                     data_batch: DATA_BATCH = None,
    #                     outputs: Optional[Sequence] = None) -> None:
    #     """All subclasses should override this method, if they need any
    #     operations after each test iteration.

    #     Args:
    #         runner (Runner): The runner of the training  process.
    #         batch_idx (int): The index of the current batch in the test loop.
    #         data_batch (dict or tuple or list, optional): Data from dataloader.
    #         outputs (Sequence, optional): Outputs from model.
    #     """
    #      # 1. 获取原图
    #     images = data_batch['inputs']  
        
        
    #     model_head = get_ori_model(runner.model).head
    #     # 2. 获取 attention map 和 mask
    #     # 假设 outputs 里有 attention/mask
    #     attention_weight = getattr(model_head.task_heads['classification'], 'attention_weights', None)
    #     attention_mask = getattr(model_head.task_heads['classification'], 'attention_masks', None)
        
    #     # 3. 保存
    #     save_attention_visualization(images, attention_weight[0], attention_mask[0], os.path.join(runner.work_dir, "attention_visualizations"),data_batch=data_batch)
@HOOKS.register_module()
class MultiLabelCausalHook(Hook):
    """Hook for SwAV.

    This hook builds the queue in SwAV according to ``epoch_queue_starts``.
    The queue will be saved in ``runner.work_dir`` or loaded at start epoch
    if the path folder has queues saved before.

    Args:
        batch_size (int): the batch size per GPU for computing.
        epoch_queue_starts (int, optional): from this epoch, starts to use the
            queue. Defaults to 15.
        crops_for_assign (list[int], optional): list of crops id used for
            computing assignments. Defaults to [0, 1].
        feat_dim (int, optional): feature dimension of output vector.
            Defaults to 128.
        queue_length (int, optional): length of the queue (0 for no queue).
            Defaults to 0.
        interval (int, optional): the interval to save the queue.
            Defaults to 1.
        frozen_layers_cfg (dict, optional): Dict to config frozen layers.
            The key-value pair is layer name and its frozen iters. If frozen,
            the layers don't need gradient. Defaults to dict().
    """

    def __init__(
        self,
        batch_size: int,
        epoch_queue_starts: Optional[int] = 15,
        crops_for_assign: Optional[List[int]] = [0, 1],
        feat_dim: Optional[int] = 128,
        queue_length: Optional[int] = 0,
        interval: Optional[int] = 1,
        frozen_epoch_cfg: Optional[Dict] = dict(),
        loss_decay: Optional[Dict] = dict()
    ) -> None:
        self.batch_size = batch_size * get_world_size()
        self.epoch_queue_starts = epoch_queue_starts
        self.crops_for_assign = crops_for_assign
        self.feat_dim = feat_dim
        self.queue_length = queue_length
        self.interval = interval
        self.frozen_epoch_cfg = frozen_epoch_cfg
        self.loss_decay =loss_decay
        self.requires_grad = True
        self.queue = None
    


    def before_run(self, runner) -> None:
        """Check whether the queues exist locally or not."""
        if is_distributed():
            self.queue_path = osp.join(runner.work_dir,
                                       'queue' + str(get_rank()) + '.pth')
        else:
            self.queue_path = osp.join(runner.work_dir, 'queue.pth')
        # load the queues if queues exist locally
        if osp.isfile(self.queue_path):
            self.queue = torch.load(self.queue_path)['queue']
            # print(runner.model)
            if hasattr(runner.model, 'module'):
                if hasattr(runner.model.module, 'module'):  # Check if wrapped by TTA
                    model = runner.model.module.module
                else:
                    model = runner.model.module
            else:  # Fallback to original model
                model = runner.model
            # print(model)
           
            get_ori_model(model).head.swav_loss_module.queue = self.queue
            MMLogger.get_current_instance().info(
                f'Load queue from file: {self.queue_path}')
        # the queue needs to be divisible by the batch size
        self.queue_length -= self.queue_length % self.batch_size
        # runner.model.train_step = Gan_train_step.__get__(runner.model)
        runner.model.memory_train_epoch = self.loss_decay.total_epochs
        runner.model.max_epochs = runner.max_epochs
        # get_ori_model(runner.model).head.memory_train_epoch = self.loss_decay.total_epochs

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: Optional[Sequence[dict]] = None) -> None:
        """Freeze layers before specific iters according to the config."""
        for layer, frozen_epoch in self.frozen_epoch_cfg.items():
            if runner.epoch <= frozen_epoch:
                for name, p in get_ori_model(runner.model).named_parameters():
                    if layer in name:
                        p.requires_grad = False
            elif runner.epoch > frozen_epoch:
                # runner.logger.info('I am using')
                if self.loss_decay:
                    if runner.epoch < self.loss_decay.total_epochs:
                        for name, p in get_ori_model(runner.model).named_parameters():
                            if layer in name:
                                p.requires_grad = True
                    else:
                        for name, p in get_ori_model(runner.model).named_parameters():
                            if layer in name:
                                p.requires_grad = False
                else:
                    for name, p in get_ori_model(runner.model).named_parameters():
                            if layer in name:
                                p.requires_grad = True

           
        get_ori_model(runner.model).head.swav_loss_module.data_sample=data_batch['data_samples']


    def before_train_epoch(self, runner) -> None:
        """Check the queues' state."""
        # optionally starts a queue
        runner.model.current_epoch = runner.epoch 
        
        if self.queue_length > 0 \
            and runner.epoch >= self.epoch_queue_starts \
                and self.queue is None:

            self.queue = torch.zeros(
                # len(self.crops_for_assign),
                self.queue_length // runner.world_size,
                self.feat_dim,
                device=get_device(),
            )
            self.label_queue = torch.zeros(
                # len(self.crops_for_assign),
                self.queue_length // runner.world_size,
                get_ori_model(runner.model).head.num_class,
                device=get_device(),
            )
            get_ori_model(runner.model).head.swav_loss_module.queue = self.queue
            get_ori_model(runner.model).head.swav_loss_module.label_queue = self.label_queue
            get_ori_model(runner.model).head.swav_loss_module.use_queue = False
        # set the boolean type of use_the_queue
        get_ori_model(runner.model).head.current_epoch = runner.epoch
        get_ori_model(runner.model).head.warmup_epoch = get_ori_model(runner.model).warmup_epoch
       

    def after_train_epoch(self, runner) -> None:
        """Save the queues locally."""
        self.queue = get_ori_model(runner.model).head.swav_loss_module.queue
        # self.label_queue = get_ori_model(runner.model).head.swav_loss_module.label_queue
        get_ori_model(runner.model).head.swav_loss_module.data_sample = None
        # if self.queue is not None and runner.epoch == self.frozen_epoch_cfg.prototypes:
        #     queue_np = self.queue.cpu().numpy()  # Convert to NumPy for k-means
        #     # label_np = self.label_queue.cpu().numpy
        #     # Apply k-means clustering
        #     num_clusters = get_ori_model(runner.model).head.swav_loss_module.prototypes.shape[0]  # Ensure valid cluster count
        #     kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
        #     cluster_assignments=kmeans.fit_predict(queue_np)
        #     cluster_assignments = torch.tensor(cluster_assignments, dtype=torch.long, device=self.queue.device)
        #     # cluster_assignments=kmeans.fit_predict(queue_np)
        #     # weights = queue_labels @ queue_labels.T.mean(dim=1)  # 标签相似度作为权重
        #     # kmeans = WeightedKMeans(n_clusters=K)
        #     # kmeans.fit(queue_features, sample_weight=weights)
        #     # new_prototypes = kmeans.cluster_centers_

        #     new_weight = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=self.queue.device)
        #     get_ori_model(runner.model).head.swav_loss_module.prototypes.data.copy_(new_weight)

        #     # class_counts = torch.zeros((num_clusters), device=self.queue.device)
        #     # for k in range(num_clusters):
        #     #     mask = (cluster_assignments == k)
        #     #     class_counts[k] = mask.sum()
        #     # # runner.logger.info(
        #     # # f"Epoch {runner.epoch} Prototype Prior - "f"{queue_np[1:7,:]}")
        #     # prior = torch.softmax(class_counts, dim=0)  #
        #     # get_ori_model(runner.model).head.swav_loss_module.prior=prior

        #     runner.logger.info("Updated loss weight using k-means clustering.")
        
        if self.loss_decay:
            if self.loss_decay.decay_type == 'linear':
                progress = min(runner.epoch / self.loss_decay.total_epochs, 1.0)
                new_weight = self.loss_decay.start_weight - (self.loss_decay.start_weight - self.loss_decay.end_weight) * progress
            elif self.loss_decay.decay_type == 'exponential':
                decay_rate = (self.loss_decay.end_weight / self.loss_decay.start_weight) ** (1 / self.loss_decay.total_epochs)
                new_weight = self.loss_decay.start_weight * (decay_rate ** runner.epoch)
            new_weight = max(self.loss_decay.end_weight, new_weight)
            # print("Before update:", get_ori_model(runner.model).head.swav_loss_module.weight)  # Debugging
            # with torch.no_grad():
            #     get_ori_model(runner.model).head.swav_loss_module.weight = new_weight
            #     runner.model.head.swav_loss_module.weight.copy_(torch.tensor(new_weight, dtype=torch.float32, device=runner.model.head.swav_loss_module.weight.device))
            get_ori_model(runner.model).head.swav_loss_module.weight = new_weight
            runner.logger.info(f"Updated loss weight to: {get_ori_model(runner.model).head.swav_loss_module.weight}") 
            # print("After update:", get_ori_model(runner.model).head.swav_loss_module.weight)  # Debugging
            # 1. 获取原型权重矩阵

        prototypes = get_ori_model(runner.model).head.swav_loss_module.prototypes
        # 2. 计算余弦相似度矩阵
        with torch.no_grad():
            # 归一化原型向量
            normalized_proto = F.normalize(prototypes, p=2, dim=1)
            
            # 计算全连接相似度矩阵
            similarity_matrix = torch.mm(normalized_proto, normalized_proto.t())
            
            # 排除对角线元素（自身相似度）
            mask = torch.eye(similarity_matrix.size(0), 
                            dtype=torch.bool, 
                            device=similarity_matrix.device)
            filtered_sim = similarity_matrix[~mask].view(
                            similarity_matrix.size(0), -1)
        
        # 3. 计算统计指标
        stats = {
            # 平均相似度（排除对角线）
            'proto/mean_sim': filtered_sim.mean().item(),
            
            # 最大相似度（排除对角线）
            'proto/max_sim': filtered_sim.max(dim=1)[0].mean().item(),
            
            # 最小相似度（排除对角线）
            'proto/min_sim': filtered_sim.min(dim=1)[0].mean().item(),
            
            # # 相似度矩阵直方图
            # 'proto/sim_hist': wandb.Histogram(similarity_matrix.cpu().numpy())
        }
        
        # 4. 记录到日志系统
        # runner.logger.info(
        #     f"Epoch {runner.epoch} Prototype Prior - "f"{self.queue[1:7,:]}")
        runner.logger.info(
            f"Epoch {runner.epoch} Prototype Prior - "f"{get_ori_model(runner.model).head.swav_loss_module.prior.squeeze().cpu().numpy()}")
        runner.logger.info(
            f"Epoch {runner.epoch} Prototype Similarity - "
            f"Mean: {stats['proto/mean_sim']:.4f}, "
            f"Max: {stats['proto/max_sim']:.4f}, "
            f"Min: {stats['proto/min_sim']:.4f}"
        ) 
        if self.queue is not None and self.every_n_epochs(
                runner, self.interval):
            torch.save({'queue': self.queue}, self.queue_path)
            queue_mean = self.queue.mean().item()
            queue_nonzero = (self.queue != 0).float().mean().item()
            runner.logger.info(f"Queue Stats - Mean: {queue_mean:.4f}, Non-zero: {queue_nonzero:.4f}")

def save_attention_visualization(images, attention_weight, attention_mask, save_dir, data_batch):
    """
    images: [B, 3, H, W] (tensor, 0-1 or 0-255)
    attention_weight: [B, C, N] (N=H_feat*W_feat)
    attention_mask: [B, C, N] (bool or float)
    save_dir: directory to save images
    prefix: filename prefix
    """
    os.makedirs(save_dir, exist_ok=True)
    B, C, N = attention_weight.shape
    _, _, H, W = images.shape
    grid_size = int(np.sqrt(N))
    assert grid_size * grid_size == N, "N must be a square number (e.g., 14*14, 7*7, etc.)"

    for b in range(B):
        fig, axes = plt.subplots(6, C+1, figsize=(3*(C+1), 18))
        image_name= data_batch['data_samples'][b].metainfo.get('img_path', f"img{b}")
        # Extract second-to-last dir, last dir, and filename
        path_parts = image_name.split(os.sep)
        second_last_dir = path_parts[-3] if len(path_parts) >= 3 else ""
        last_dir = path_parts[-2] if len(path_parts) >= 2 else ""
        filename = os.path.splitext(os.path.basename(image_name))[0]
        gt_labels = data_batch['data_samples'][b].gt_score.cpu().numpy()
        gt_score = "_".join([str(int(x)) for x in gt_labels])
        prefix = f"{second_last_dir}_{last_dir}_{filename}_gt_{gt_score}"
        img = images[b].cpu()
        if img.max() > 1.1:
            img = img / 255.0
        img_np = np.transpose(img.numpy(), (1, 2, 0))
        # 第一行：原图 + attention map
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        import matplotlib.cm as cm
        # 自然柔和的配色（高区分度）
        base_colors = [
            (0.85, 0.37, 0.31),  # 珊瑚红
            (0.40, 0.65, 0.82),  # 天空蓝
            (0.55, 0.76, 0.29),  # 草绿色
            (0.91, 0.74, 0.42),  # 金黄色
            (0.69, 0.51, 0.82),  # 薰衣草紫
            (0.45, 0.77, 0.69),  # 青绿色
            (0.96, 0.60, 0.62),  # 粉红色
            (0.70, 0.54, 0.42),  # 褐色
            (0.51, 0.66, 0.88),  # 浅蓝色
            (0.89, 0.68, 0.31),  # 橙黄色
        ]
        for k in range(C):
            base_color = base_colors[k % len(base_colors)]
            base_img = img_np * 0.6 + np.array(base_color)[None, None, :] * 0.4
            base_img = np.clip(base_img, 0, 1)
            # AttnMap
            attn_map = attention_weight[b, k].reshape(grid_size, grid_size).cpu().numpy()
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            attn_map = np.clip(attn_map, 0, 1)
            attn_map = np.array(Image.fromarray((attn_map*255).astype(np.uint8)).resize((W, H), resample=Image.BILINEAR))/255.0
            # AttnMap
            axes[0, k+1].imshow(base_img)
            # 只显示高激活区域的jet色（设置阈值，低于阈值的区域透明）
            attn_map_masked = np.ma.masked_where(attn_map < 0.3, attn_map)
            axes[0, k+1].imshow(attn_map_masked, cmap='jet', alpha=0.7, vmin=0, vmax=1)
            axes[0, k+1].set_title(f'AttnMap-{k}')
            axes[0, k+1].axis('off')
            # AttnMask: mask区域内显示黑色，mask区域外显示底色+高亮的attention map
            mask_map = attention_mask[b, k].reshape(grid_size, grid_size).cpu().numpy()
            mask_map = mask_map.astype(float)
            mask_map = np.array(Image.fromarray((mask_map*255).astype(np.uint8)).resize((W, H), resample=Image.NEAREST))/255.0
            mask_black_img = base_img.copy()
            mask_binary = mask_map > 0.5
            mask_black_img[mask_binary] = 0  # mask区域显示黑色
            axes[1, k+1].imshow(mask_black_img)
            # 在非mask区域显示高激活区域的attention map
            attn_non_mask_high = np.ma.masked_where((mask_map >= 0.5) | (attn_map < 0.3), attn_map)
            axes[1, k+1].imshow(attn_non_mask_high, cmap='jet', alpha=0.7, vmin=0, vmax=1)
            axes[1, k+1].set_title(f'AttnMask-{k}')
            axes[1, k+1].axis('off')
            # OtherMinusSelf: mask区域内显示黑色，mask区域外显示底色+attention map
            others = [i for i in range(C) if i != k]
            if len(others) > 0:
                others_union_mask = attention_mask[b, others].any(axis=0)  # 其他疾病的并集
                self_mask = attention_mask[b, k].bool() if attention_mask.dtype != torch.bool else attention_mask[b, k]
                diff_mask = others_union_mask & (~self_mask)  # 其他疾病的mask去除当前疾病mask
            else:
                diff_mask = torch.zeros_like(attention_mask[b, k]).cpu()
            diff_mask_map = diff_mask.reshape(grid_size, grid_size).cpu().numpy().astype(float)
            diff_mask_map = np.array(Image.fromarray((diff_mask_map*255).astype(np.uint8)).resize((W, H), resample=Image.NEAREST))/255.0
            diff_black_img = base_img.copy()
            diff_binary = diff_mask_map > 0.5
            diff_black_img[diff_binary] = 0  # mask区域显示黑色
            axes[2, k+1].imshow(diff_black_img)
            # 在非mask区域显示高激活区域的attention map
            attn_non_diff_high = np.ma.masked_where((diff_mask_map >= 0.5) | (attn_map < 0.3), attn_map)
            axes[2, k+1].imshow(attn_non_diff_high, cmap='jet', alpha=0.7, vmin=0, vmax=1)
            axes[2, k+1].set_title(f'OthMinusSelf-{k}')
            axes[2, k+1].axis('off')
            
            # 第四行：AttnMask区域显示为黑色，非mask区域底色占主导+淡化的attention map
            mask_black = base_img.copy()
            mask_binary = mask_map > 0.5
            mask_black[mask_binary] = 0  # mask区域设为黑色
            axes[3, k+1].imshow(mask_black)
            # 在非mask区域叠加淡化的attention map（alpha=0.3）
            attn_non_mask_light = np.ma.masked_where(mask_map >= 0.5, attn_map)
            axes[3, k+1].imshow(attn_non_mask_light, cmap='jet', alpha=0.3, vmin=0, vmax=1)
            axes[3, k+1].set_title(f'AttnMask-Black-{k}')
            axes[3, k+1].axis('off')
            
            # 第五行：OtherMinusSelf区域显示为黑色，非mask区域底色占主导+淡化的attention map
            diff_black = base_img.copy()
            diff_binary = diff_mask_map > 0.5
            diff_black[diff_binary] = 0  # mask区域设为黑色
            axes[4, k+1].imshow(diff_black)
            # 在非mask区域叠加淡化的attention map（alpha=0.3）
            attn_non_diff_light = np.ma.masked_where(diff_mask_map >= 0.5, attn_map)
            axes[4, k+1].imshow(attn_non_diff_light, cmap='jet', alpha=0.3, vmin=0, vmax=1)
            axes[4, k+1].set_title(f'OthMinusSelf-Black-{k}')
            axes[4, k+1].axis('off')
            
            # 第六行：mask区域内显示原图+attention map，mask区域外用黑色覆盖
            mask_region_img = img_np.copy()
            mask_binary = mask_map > 0.5
            mask_region_img[~mask_binary] = 0  # mask外区域设为黑色
            axes[5, k+1].imshow(mask_region_img)
            # 在mask内显示attention map
            attn_inside_mask = np.ma.masked_where(mask_map < 0.5, attn_map)
            axes[5, k+1].imshow(attn_inside_mask, cmap='jet', alpha=0.7, vmin=0, vmax=1)
            axes[5, k+1].set_title(f'MaskOnly-{k}')
            axes[5, k+1].axis('off')
        
        # 所有行的第一列：原图
        for row in range(6):
            axes[row, 0].imshow(img_np)
            axes[row, 0].set_title('Original')
            axes[row, 0].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{prefix}_img{b}.png")
        plt.savefig(save_path)
        plt.close(fig)

# 用法示例：
# save_attention_visualization(images, attention_weight, attention_mask, save_dir="./vis", prefix="test")
