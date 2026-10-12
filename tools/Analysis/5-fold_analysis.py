# ========== 统计cf/margin二维表格并输出markdown ===========

# 用法示例：
# generate_cf_margin_tables('work_dirs_MICCAI2026/ODIR/CounterFactual', 'ODIR')
# generate_cf_margin_tables('work_dirs_MICCAI2026/Endo/CounterFactual', 'Endo')
# === 自动处理所有CounterFactual实验目录下TDE统计 ===
import re
import os
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import re

# =====================
# main函数统一入口

def extract_param_from_folder(folder_name):
    """
    从文件夹名称中解析超参数。

    支持的命名格式（按优先级）：
    1) cf{cf}_margin{margin}_lpos{lpos}_lneg{lneg}
       → 用于 lpos / lneg 网格搜索
    2) cf{cf}_margin{margin}_lambda{lambda}
       → 旧版单一 lambda 网格
    3) cf{cf}_margin{margin}

    返回值：cf, margin, lambda_val, lpos, lneg
    如果某个字段不存在则为 None。
    """
    # More flexible parsing: search for each token anywhere in the folder name.
    # This lets us handle folder names like
    #   cf0.4_margin0.4_lpos0.1_lneg0.3
    #   lpos0.9_lneg0.9
    #   cf0.4_margin0.4
    cf = None
    margin = None
    lambda_val = None
    lpos = None
    lneg = None

    m = re.search(r'cf([\d\.]+)', folder_name)
    if m:
        try:
            cf = float(m.group(1))
        except Exception:
            cf = None
    m = re.search(r'margin([\d\.]+)', folder_name)
    if m:
        try:
            margin = float(m.group(1))
        except Exception:
            margin = None
    m = re.search(r'lambda([\d\.]+)', folder_name)
    if m:
        try:
            lambda_val = float(m.group(1))
        except Exception:
            lambda_val = None
    m = re.search(r'lpos([\d\.]+)', folder_name)
    if m:
        try:
            lpos = float(m.group(1))
        except Exception:
            lpos = None
    m = re.search(r'lneg([\d\.]+)', folder_name)
    if m:
        try:
            lneg = float(m.group(1))
        except Exception:
            lneg = None

    return cf, margin, lambda_val, lpos, lneg

def extract_mean_mauc(md_path):
    if not os.path.exists(md_path):
        return None, None
    in_block = False
    for line in open(md_path, 'r'):
        if line.strip().startswith('## multi-class/mAUC_multiclass per fold'):
            in_block = True
            continue
        if in_block:
            if line.strip().startswith('## '):  # 下一个表格块开始
                break
            if '| **Mean±Std** | **' in line:
                m = re.search(r'\*\*([\d\.eE+-]+)±([\d\.eE+-]+)\*\*', line)
                if m:
                    try:
                        mean = float(m.group(1))
                        std = float(m.group(2))
                        return mean, std
                    except Exception:
                        pass
    return None, None

def extract_tde_value(md_path, key):
    # 只在# 5-Fold Mean Statistics区块内提取
    if not os.path.exists(md_path):
        return None, None
    in_block = False
    with open(md_path, 'r') as f:
        for line in f:
            if line.strip().startswith('# 5-Fold Mean Statistics'):
                in_block = True
                continue
            if in_block:
                if line.strip().startswith('#') and not line.strip().startswith('# 5-Fold Mean Statistics'):
                    break
                # 支持key参数带或不带(mean+std)
                if f'- **{key} (mean+std):**' in line or f'- **{key}:**' in line:
                    m = re.search(r'- \*\*' + re.escape(key) + r'(?: \(mean\+std\))?:\*\*\s*([\-\d\.eEinf]+)\+([\-\d\.eEinf]+)', line)
                    if m:
                        mean_val = m.group(1)
                        std_val = m.group(2)
                        try:
                            mean_val = float(mean_val) if mean_val != 'inf' else float('inf')
                            std_val = float(std_val) if std_val != 'inf' else float('inf')
                            return mean_val, std_val
                        except Exception:
                            return None, None
                    # 兼容只写mean的情况
                    m2 = re.search(r'- \*\*' + re.escape(key) + r'(?: \(mean\+std\))?:\*\*\s*([\-\d\.eEinf]+)', line)
                    if m2:
                        mean_val = m2.group(1)
                        try:
                            mean_val = float(mean_val) if mean_val != 'inf' else float('inf')
                            return mean_val, None
                        except Exception:
                            return None, None
    return None, None


def generate_cf_margin_tables(base_dir, dataset_name, output_md_path=None):
    """
    智能生成参数网格表格：
    - 支持 cf×margin 网格（传统）
    - 支持 lpos×lneg 网格（lambda ablation）
    - 支持 lambda 一维展开
    - 自动从父目录名提取固定参数
    """
    indicators = ['mean mAUC', 'Overall OR Median', 'Overall TDE Negative', 'Overall TDE Positive']
    
    # ========== 辅助函数: 格式化指标值 ==========
    def _format_value(indicator, val):
        if indicator == 'mean mAUC':
            if val is None or val[0] is None or val[1] is None:
                return ''
            return f'{val[0]:.4f}±{val[1]:.4f}'
        if val is None or val[0] is None:
            return ''
        if val[0] == float('inf'):
            return 'inf'
        if val[1] is not None:
            return f'{val[0]:.4f}+{val[1]:.4f}'
        return f'{val[0]:.4f}'
    
    # ========== 阶段1: 扫描所有子文件夹，收集数据 ==========
    all_data = {}  # {folder_name: {indicator: (mean, std)}}
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        mauc_mean, mauc_std = extract_mean_mauc(os.path.join(folder_path, '5fold_results.md'))
        or_median_mean, or_median_std = extract_tde_value(os.path.join(folder_path, 'tde_summary.md'), 'Overall OR Median')
        tde_neg_mean, tde_neg_std = extract_tde_value(os.path.join(folder_path, 'tde_summary.md'), 'Overall TDE Negative')
        tde_pos_mean, tde_pos_std = extract_tde_value(os.path.join(folder_path, 'tde_summary.md'), 'Overall TDE Positive')
        all_data[folder] = {
            'mean mAUC': (mauc_mean, mauc_std),
            'Overall OR Median': (or_median_mean, or_median_std),
            'Overall TDE Negative': (tde_neg_mean, tde_neg_std),
            'Overall TDE Positive': (tde_pos_mean, tde_pos_std)
        }
    
    if not all_data:
        print(f'No experiment folders found in {base_dir}')
        return
    
    # ========== 阶段2: 从父目录名提取固定参数 ==========
    base_name = os.path.basename(base_dir.rstrip(os.sep))
    base_cf, base_margin, base_lambda, base_lpos, base_lneg = extract_param_from_folder(base_name)
    print(f"Base directory: {base_name}")
    print(f"  Fixed params from base_dir: cf={base_cf}, margin={base_margin}, lambda={base_lambda}, lpos={base_lpos}, lneg={base_lneg}")
    
    # ========== 阶段3: 解析每个子文件夹的参数，合并固定参数 ==========
    results = {}  # {(cf, margin, lambda, lpos, lneg): data}
    cf_set, margin_set, lambda_set, lpos_set, lneg_set = set(), set(), set(), set(), set()
    
    for folder, data in all_data.items():
        # 从子文件夹名解析参数
        cf, margin, lambda_val, lpos, lneg = extract_param_from_folder(folder)
        
        # 如果子文件夹没有某参数，使用父目录的固定值
        if cf is None:
            cf = base_cf
        if margin is None:
            margin = base_margin
        if lambda_val is None:
            lambda_val = base_lambda
        if lpos is None:
            lpos = base_lpos
        if lneg is None:
            lneg = base_lneg
        
        # 收集所有非 None 的参数值
        if cf is not None:
            cf_set.add(cf)
        if margin is not None:
            margin_set.add(margin)
        if lambda_val is not None:
            lambda_set.add(lambda_val)
        if lpos is not None:
            lpos_set.add(lpos)
        if lneg is not None:
            lneg_set.add(lneg)
        
        # 存储结果（即使某些参数是 None）
        results[(cf, margin, lambda_val, lpos, lneg)] = data
    
    cf_list = sorted(cf_set)
    margin_list = sorted(margin_set)
    lambda_list = sorted(lambda_set)
    lpos_list = sorted(lpos_set)
    lneg_list = sorted(lneg_set)
    
    print(f"Parsed params: {len(cf_list)} cf, {len(margin_list)} margin, {len(lambda_list)} lambda, {len(lpos_list)} lpos, {len(lneg_list)} lneg")
    if cf_list:
        print(f"  cf values: {cf_list[:5]}")
    if margin_list:
        print(f"  margin values: {margin_list[:5]}")
    if lambda_list:
        print(f"  lambda values: {lambda_list[:5]}")
    if lpos_list:
        print(f"  lpos values: {lpos_list[:5]}")
    if lneg_list:
        print(f"  lneg values: {lneg_list[:5]}")
    
    # ========== 阶段4: 决定展开策略 ==========
    # 优先级：lpos×lneg > lambda 一维 > cf×margin
    if len(lpos_list) > 1 and len(lneg_list) > 1:
        # lpos×lneg 网格（cf/margin 固定或不存在）
        mode = 'lpos_lneg'
        row_label = 'lneg\\lpos'
        row_params = lneg_list
        col_params = lpos_list
        fixed_cf = cf_list[0] if len(cf_list) == 1 else None
        fixed_margin = margin_list[0] if len(margin_list) == 1 else None
        print(f"→ Mode: lpos×lneg grid (cf={fixed_cf}, margin={fixed_margin})")
    elif len(lambda_list) > 1 and len(cf_list) == 1 and len(margin_list) == 1:
        # lambda 一维展开（cf/margin 固定）
        mode = 'lambda'
        row_label = 'cf\\lambda'
        row_params = [cf_list[0]]
        col_params = lambda_list
        print(f"→ Mode: lambda sweep (cf={cf_list[0]}, margin={margin_list[0]})")
    elif len(cf_list) > 0 and len(margin_list) > 0:
        # cf×margin 网格（传统）
        mode = 'cf_margin'
        row_label = 'cf\\margin'
        row_params = cf_list
        col_params = margin_list
        print(f"→ Mode: cf×margin grid")
    else:
        # 无法结构化，输出简单表格
        print(f"→ Mode: fallback (folder summary)")
        folder_names = sorted(all_data.keys())
        md_lines = [f'\n### {dataset_name} - Summary by Folder\n']
        header = ['Indicator'] + folder_names
        md_lines.append('| ' + ' | '.join(header) + ' |')
        md_lines.append('|' + '---|' * len(header))
        for indicator in indicators:
            row = [indicator]
            for folder in folder_names:
                val = all_data[folder].get(indicator)
                row.append(_format_value(indicator, val))
            md_lines.append('| ' + ' | '.join(row) + ' |')
        print('\n'.join(md_lines))
        if output_md_path:
            with open(output_md_path, 'w') as f:
                f.write('\n'.join(md_lines))
        return
    
    # ========== 阶段5: 生成表格 ==========
    md_lines = []
    for indicator in indicators:
        md_lines.append(f'\n### {dataset_name} - {indicator}\n')
        
        # 生成表头
        header = [row_label] + [f'{p:.2f}' for p in col_params]
        md_lines.append('| ' + ' | '.join(header) + ' |')
        md_lines.append('|' + '---|' * len(header))
        
        # 生成数据行
        if mode == 'lpos_lneg':
            # lpos×lneg 网格：行=lneg，列=lpos
            for lneg in row_params:
                row = [f'{lneg:.2f}']
                for lpos in col_params:
                    # 查找对应的结果（cf/margin 可能为 None）
                    val = None
                    for key, data in results.items():
                        key_cf, key_margin, key_lambda, key_lpos, key_lneg = key
                        if key_lpos == lpos and key_lneg == lneg:
                            # 如果有固定的 cf/margin，检查是否匹配
                            if (fixed_cf is None or key_cf == fixed_cf) and \
                               (fixed_margin is None or key_margin == fixed_margin):
                                val = data.get(indicator)
                                break
                    row.append(_format_value(indicator, val))
                md_lines.append('| ' + ' | '.join(row) + ' |')
        
        elif mode == 'lambda':
            # lambda 一维展开：行=cf，列=lambda
            for cf in row_params:
                row = [f'{cf:.2f}']
                for lambda_val in col_params:
                    val = results.get((cf, margin_list[0], lambda_val, None, None), {}).get(indicator)
                    row.append(_format_value(indicator, val))
                md_lines.append('| ' + ' | '.join(row) + ' |')
        
        else:  # mode == 'cf_margin'
            # cf×margin 网格：行=cf，列=margin
            for cf in row_params:
                row = [f'{cf:.2f}']
                for margin in col_params:
                    # 查找匹配的结果（忽略 lambda/lpos/lneg）
                    val = None
                    for key, data in results.items():
                        key_cf, key_margin, _, _, _ = key
                        if key_cf == cf and key_margin == margin:
                            val = data.get(indicator)
                            break
                    row.append(_format_value(indicator, val))
                md_lines.append('| ' + ' | '.join(row) + ' |')
        
        # 控制台输出当前表格
        num_rows = len(row_params)
        print('\n'.join(md_lines[-(num_rows+3):]))
    
    # 保存主表格
    if output_md_path:
        with open(output_md_path, 'w') as f:
            f.write('\n'.join(md_lines))
        print(f"\n✓ Main tables saved to: {output_md_path}")
    
    # ========== 阶段6: 生成 per-(cf,margin) 的 lpos×lneg 表格 ==========
    # 如果存在多个 (cf, margin) 组合且都有 lpos/lneg 变化，为每个组合生成独立表格
    if len(lpos_list) > 0 and len(lneg_list) > 0 and mode != 'lpos_lneg':
        # 收集所有包含 lpos/lneg 的 (cf, margin) 组合
        cf_margin_with_lpos = {}
        for key, data in results.items():
            cf_k, margin_k, _, lpos_k, lneg_k = key
            if lpos_k is not None and lneg_k is not None and cf_k is not None and margin_k is not None:
                cf_margin_with_lpos.setdefault((cf_k, margin_k), set()).add((lpos_k, lneg_k))
        
        if cf_margin_with_lpos:
            base_out_dir = os.path.dirname(output_md_path) if output_md_path else base_dir
            for (cf_k, margin_k), pairs in sorted(cf_margin_with_lpos.items()):
                lpos_vals = sorted({p[0] for p in pairs})
                lneg_vals = sorted({p[1] for p in pairs})
                
                table_lines = [f"\n### {dataset_name} - lpos×lneg (cf={cf_k:.2f}, margin={margin_k:.2f})\n"]
                header = ['lneg\\lpos'] + [f'{lp:.2f}' for lp in lpos_vals]
                table_lines.append('| ' + ' | '.join(header) + ' |')
                table_lines.append('|' + '---|' * len(header))
                
                for lneg in lneg_vals:
                    row = [f'{lneg:.2f}']
                    for lpos in lpos_vals:
                        val = results.get((cf_k, margin_k, None, lpos, lneg), {}).get('mean mAUC')
                        row.append(_format_value('mean mAUC', val))
                    table_lines.append('| ' + ' | '.join(row) + ' |')
                
                out_name = os.path.join(base_out_dir, f'{dataset_name}_cf{cf_k:.2f}_margin{margin_k:.2f}_lpos_lneg.md')
                with open(out_name, 'w') as f:
                    f.write('\n'.join(table_lines))
                print(f"✓ Per-(cf,margin) lpos×lneg table: {out_name}")


def process_classification_metrics(base_root_dir):
    """
    处理原有的分类/多标签等常规指标统计与写入。
    """
    all_exp_dirs = [os.path.join(base_root_dir, d) for d in os.listdir(base_root_dir) if os.path.isdir(os.path.join(base_root_dir, d))]
    for base_fold_dir in all_exp_dirs:
        print(f'Processing {base_fold_dir}')
        fold_dirs = []
        for i in range(0, 5):
            fold_path = os.path.join(base_fold_dir, f'fold{i}')
            json_files = glob.glob(os.path.join(fold_path, '**', '*.json'), recursive=True)
            if json_files:
                json_files.sort()
                fold_dirs.append(json_files[-1])
            else:
                print(f'Warning: No json file found in {fold_path}')

        output_dir = base_fold_dir
        os.makedirs(output_dir, exist_ok=True)
        for text in ['TDE_', 'Causal_','']:
            multi_class_auc = []
            multi_class_cohen_kappa = []
            multi_class_quadratic_weighted_kappa = []
            multi_class_auc_score = []
            multi_class_confusion_matrix = []
            multi_class_accuracy = []
            multi_class_top1 = []
            multi_class_precision = []
            multi_class_recall = []
            multi_class_f1 = []
            multi_label_metrics = {'precision': [], 'recall': [], 'f1-score': []}
            multi_label_metrics_classwise = {'precision': [], 'recall': [], 'f1-score': []}
            roc_curves_fpr = []
            roc_curves_tpr = []
            multi_label_auc_curves = []
            multi_class_mAUC_multiclass = []
            multi_label_mAUC = []
            multi_label_mAP = []

            for json_path in fold_dirs:
                with open(json_path, 'r') as f:
                    result = json.load(f)

                def safe_append(lst, key, alt_key=None):
                    if key in result:
                        lst.append(result[key])
                    elif alt_key and alt_key in result:
                        lst.append(result[alt_key])

                safe_append(multi_class_cohen_kappa, f'multi-class/{text}cohen_kappa')
                safe_append(multi_class_quadratic_weighted_kappa, f'multi-class/{text}quadratic_weighted_kappa')
                safe_append(multi_class_auc_score, f'multi-class/{text}auc')
                safe_append(multi_class_confusion_matrix, f'multi-class/{text}confusion_matrix')
                safe_append(multi_class_auc, f'multi-class/{text}mAUC_multi_class')
                safe_append(multi_class_precision, f'multi-class/{text}precision')
                safe_append(multi_class_recall, f'multi-class/{text}recall')
                safe_append(multi_class_f1, f'multi-class/{text}f1-score', alt_key=f'multi-class/{text}f1_score')
                safe_append(multi_class_accuracy, f'multi-label/{text}accuracy')
                safe_append(multi_class_top1, f'multi-class/{text}top1')
                safe_append(multi_label_metrics['precision'], f'multi-label/{text}precision')
                safe_append(multi_label_metrics['recall'], f'multi-label/{text}recall')
                safe_append(multi_label_metrics['f1-score'], f'multi-label/{text}f1-score', alt_key=f'multi-label/{text}f1_score')
                safe_append(multi_label_metrics_classwise['precision'], f'multi-label/{text}precision_classwise')
                safe_append(multi_label_metrics_classwise['recall'], f'multi-label/{text}recall_classwise')
                safe_append(multi_label_metrics_classwise['f1-score'], f'multi-label/{text}f1-score_classwise', alt_key=f'multi-label/{text}f1_score_classwise')
                safe_append(roc_curves_tpr, f'multi-class/{text}tpr_list')
                safe_append(roc_curves_fpr, f'multi-class/{text}fpr_list')
                if 'multi-label/auc_curves' in result:
                    multi_label_auc_curves.append(result['multi-label/auc_curves'])
                safe_append(multi_class_mAUC_multiclass, f'multi-class/{text}mAUC_multiclass')
                safe_append(multi_label_mAUC, f'multi-label/{text}mAUC')
                safe_append(multi_label_mAP, f'multi-label/{text}mAP')

            num_folds = max(
                len(multi_class_precision),
                len(multi_class_recall),
                len(multi_class_f1),
                len(multi_class_accuracy),
                len(multi_class_top1),
                len(multi_class_auc),
                len(multi_class_cohen_kappa),
                len(multi_class_quadratic_weighted_kappa),
                len(multi_class_auc_score),
                len(multi_label_metrics['precision']),
                len(multi_label_metrics['recall']),
                len(multi_label_metrics['f1-score']),
                len(roc_curves_tpr),
                1
            )
            num_classes = len(roc_curves_tpr[0]) if roc_curves_tpr else 0
            mean_fpr = np.linspace(0, 1, 100)

            # ROC曲线
            if roc_curves_tpr and roc_curves_fpr:
                plt.figure(figsize=(8,6))
                auc_list = []
                std_list = []
                for cls_idx in range(num_classes):
                    tprs = []
                    for fold_idx in range(len(roc_curves_tpr)):
                        fpr = np.array(roc_curves_fpr[fold_idx][cls_idx])
                        tpr = np.array(roc_curves_tpr[fold_idx][cls_idx])
                        interp_tpr = np.interp(mean_fpr, fpr, tpr)
                        tprs.append(interp_tpr)
                    mean_tpr = np.mean(tprs, axis=0)
                    std_tpr = np.std(tprs, axis=0)
                    auc = np.trapz(mean_tpr, mean_fpr)
                    auc_list.append(auc)
                    std_list.append(np.std([np.trapz(tpr, mean_fpr) for tpr in tprs]))
                    plt.plot(mean_fpr, mean_tpr, label=f'Class {cls_idx+1} (AUC={auc:.3f}±{std_list[-1]:.3f})')
                    plt.fill_between(mean_fpr, mean_tpr-std_tpr, mean_tpr+std_tpr, alpha=0.2)
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'{text}Mean ROC Curve (5-fold, macro-average, per class)')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{text}mean_roc_auc_5fold_perclass.png'))
                plt.show()
            else:
                auc_list = []
                std_list = []

            md_path = os.path.join(output_dir, f'{text}5fold_results.md')
            def mean_std(arr):
                arr = np.array(arr)
                return np.mean(arr), np.std(arr)

            with open(md_path, 'w') as md:
                if multi_class_mAUC_multiclass:
                    mean_val, std_val = mean_std(multi_class_mAUC_multiclass)
                    md.write('## multi-class/mAUC_multiclass per fold\n')
                    md.write('| LOOP | mAUC_multiclass |\n')
                    md.write('|------|-----------------|\n')
                    for i, v in enumerate(multi_class_mAUC_multiclass):
                        md.write(f'| {i+1} | {v:.4f} |\n')
                    md.write(f'| **Mean±Std** | **{mean_val:.4f}±{std_val:.4f}** |\n\n')

                if multi_label_mAUC:
                    mean_val, std_val = mean_std(multi_label_mAUC)
                    md.write('## multi-label/mAUC per fold\n')
                    md.write('| LOOP | mAUC |\n')
                    md.write('|------|------|\n')
                    for i, v in enumerate(multi_label_mAUC):
                        md.write(f'| {i+1} | {v:.4f} |\n')
                    md.write(f'| **Mean±Std** | **{mean_val:.4f}±{std_val:.4f}** |\n\n')

                if multi_label_mAP:
                    mean_val, std_val = mean_std(multi_label_mAP)
                    md.write('## multi-label/mAP per fold\n')
                    md.write('| LOOP | mAP |\n')
                    md.write('|------|------|\n')
                    for i, v in enumerate(multi_label_mAP):
                        md.write(f'| {i+1} | {v:.2f} |\n')
                    md.write(f'| **Mean±Std** | **{mean_val:.2f}±{std_val:.2f}** |\n\n')
                # Multi-class Cohen Kappa
                if multi_class_cohen_kappa:
                    kappa_mean, kappa_std = mean_std(multi_class_cohen_kappa)
                    md.write('## Multi-class Cohen Kappa per fold\n')
                    md.write('| Fold | Cohen Kappa |\n')
                    md.write('|------|-------------|\n')
                    for i, v in enumerate(multi_class_cohen_kappa):
                        md.write(f'| {i+1} | {v:.3f} |\n')
                    md.write(f'| **Mean±Std** | **{kappa_mean:.3f}±{kappa_std:.3f}** |\n\n')

                # Multi-class Quadratic Weighted Kappa
                if multi_class_quadratic_weighted_kappa:
                    qwk_mean, qwk_std = mean_std(multi_class_quadratic_weighted_kappa)
                    md.write('## Multi-class Quadratic Weighted Kappa per fold\n')
                    md.write('| Fold | QWK |\n')
                    md.write('|------|-----|\n')
                    for i, v in enumerate(multi_class_quadratic_weighted_kappa):
                        md.write(f'| {i+1} | {v:.3f} |\n')
                    md.write(f'| **Mean±Std** | **{qwk_mean:.3f}±{qwk_std:.3f}** |\n\n')

                # Multi-class AUC
                if multi_class_auc_score:
                    auc_mean, auc_std = mean_std(multi_class_auc_score)
                    md.write('## Multi-class AUC per fold\n')
                    md.write('| Fold | AUC |\n')
                    md.write('|------|-----|\n')
                    for i, v in enumerate(multi_class_auc_score):
                        md.write(f'| {i+1} | {v:.3f} |\n')
                    md.write(f'| **Mean±Std** | **{auc_mean:.3f}±{auc_std:.3f}** |\n\n')

                # Multi-class Confusion Matrix
                if multi_class_confusion_matrix:
                    md.write('## Multi-class Confusion Matrix per fold\n')
                    for i, mat in enumerate(multi_class_confusion_matrix):
                        md.write(f'### Fold {i+1}\n')
                        md.write(str(mat) + '\n')

                # 动态表格，按每个指标实际长度生成
                header = ['Fold']
                metrics = []
                metric_data = []
                mean_std_row = ['**Mean±Std**']
                if multi_class_precision:
                    metrics.append('Precision')
                    metric_data.append(multi_class_precision)
                    prec_mean, prec_std = mean_std(multi_class_precision)
                    mean_std_row.append(f'**{prec_mean:.3f}±{prec_std:.3f}**')
                if multi_class_recall:
                    metrics.append('Recall')
                    metric_data.append(multi_class_recall)
                    recall_mean, recall_std = mean_std(multi_class_recall)
                    mean_std_row.append(f'**{recall_mean:.3f}±{recall_std:.3f}**')
                if multi_class_f1:
                    metrics.append('F1-score')
                    metric_data.append(multi_class_f1)
                    f1_mean, f1_std = mean_std(multi_class_f1)
                    mean_std_row.append(f'**{f1_mean:.3f}±{f1_std:.3f}**')
                if multi_class_accuracy:
                    metrics.append('Accuracy')
                    metric_data.append(multi_class_accuracy)
                    acc_mean, acc_std = mean_std(multi_class_accuracy)
                    mean_std_row.append(f'**{acc_mean:.3f}±{acc_std:.3f}**')
                if multi_class_top1:
                    metrics.append('Top1')
                    metric_data.append(multi_class_top1)
                    top1_mean, top1_std = mean_std(multi_class_top1)
                    mean_std_row.append(f'**{top1_mean:.3f}±{top1_std:.3f}**')
                if metric_data:
                    max_folds = max(len(arr) for arr in metric_data)
                    header += metrics
                    md.write('## Multi-class Metrics per fold\n')
                    md.write('| ' + ' | '.join(header) + ' |\n')
                    md.write('| ' + ' | '.join([':------:']*len(header)) + ' |\n')
                    for i in range(max_folds):
                        row = [str(i+1)]
                        for arr in metric_data:
                            if i < len(arr):
                                row.append(f'{arr[i]:.3f}')
                            else:
                                row.append('-')
                        md.write('| ' + ' | '.join(row) + ' |\n')
                    md.write('| ' + ' | '.join(mean_std_row) + ' |\n\n')

                # Multi-class mAUC
                if multi_class_auc:
                    mauc_mean, mauc_std = mean_std(multi_class_auc)
                    md.write('## Multi-class mAUC per fold\n')
                    md.write('| Fold | mAUC |\n')
                    md.write('|------|------|\n')
                    for i, v in enumerate(multi_class_auc):
                        md.write(f'| {i+1} | {v:.3f} |\n')
                    md.write(f'| **Mean±Std** | **{mauc_mean:.3f}±{mauc_std:.3f}** |\n\n')

                # Multi-label precision
                if multi_label_metrics['precision']:
                    prec_mean, prec_std = mean_std(multi_label_metrics['precision'])
                    md.write('## Multi-label precision per fold\n')
                    md.write('| LOOP | Precision |\n')
                    md.write('|------|-----------|\n')
                    for i, v in enumerate(multi_label_metrics['precision']):
                        md.write(f'| {i+1} | {v:.2f} |\n')
                    md.write(f'| **Mean±Std** | **{prec_mean:.2f}±{prec_std:.2f}** |\n\n')

                # Multi-label recall
                if multi_label_metrics['recall']:
                    recall_mean, recall_std = mean_std(multi_label_metrics['recall'])
                    md.write('## Multi-label recall per fold\n')
                    md.write('| LOOP | Recall |\n')
                    md.write('|------|--------|\n')
                    for i, v in enumerate(multi_label_metrics['recall']):
                        md.write(f'| {i+1} | {v:.2f} |\n')
                    md.write(f'| **Mean±Std** | **{recall_mean:.2f}±{recall_std:.2f}** |\n\n')

                # Multi-label f1-score
                if multi_label_metrics['f1-score']:
                    f1_mean, f1_std = mean_std(multi_label_metrics['f1-score'])
                    md.write('## Multi-label f1-score per fold\n')
                    md.write('| LOOP | F1-score |\n')
                    md.write('|------|----------|\n')
                    for i, v in enumerate(multi_label_metrics['f1-score']):
                        md.write(f'| {i+1} | {v:.2f} |\n')
                    md.write(f'| **Mean±Std** | **{f1_mean:.2f}±{f1_std:.2f}** |\n\n')

                # Multi-label classwise metrics
                if multi_label_metrics_classwise['precision']:
                    md.write('## Multi-label precision classwise per fold\n')
                    md.write(str(multi_label_metrics_classwise['precision']) + '\n\n')
                if multi_label_metrics_classwise['recall']:
                    md.write('## Multi-label recall classwise per fold\n')
                    md.write(str(multi_label_metrics_classwise['recall']) + '\n\n')
                if multi_label_metrics_classwise['f1-score']:
                    md.write('## Multi-label f1-score classwise per fold\n')
                    md.write(str(multi_label_metrics_classwise['f1-score']) + '\n\n')

                # Mean ROC Curve (per class)
                if auc_list and std_list:
                    md.write('## Mean ROC Curve (per class)\n')
                    for cls_idx, (auc, std) in enumerate(zip(auc_list, std_list)):
                        md.write(f'- Class {cls_idx+1}: AUC={auc:.3f} ± {std:.3f}\n')


def parse_tde_stats(md_lines):
    """解析tde_stats_epoch_0.md内容，返回字典"""
    stats = {}

    def to_float(token):
        token = token.strip()
        if not token:
            raise ValueError
        lower = token.lower()
        if lower in ('inf', '+inf'):
            return float('inf')
        if lower == '-inf':
            return float('-inf')
        if lower == 'nan':
            return float('nan')
        return float(token)

    i = 0
    while i < len(md_lines):
        line = md_lines[i]
        m_arr = re.match(r"\s*- \*\*(.+?):\*\*\s*(\[.*)", line)
        if m_arr:
            key = m_arr.group(1)
            buffer = m_arr.group(2).strip()
            bracket_depth = buffer.count('[') - buffer.count(']')
            while bracket_depth > 0 and i + 1 < len(md_lines):
                i += 1
                next_line = md_lines[i].strip()
                buffer += ' ' + next_line
                bracket_depth += next_line.count('[') - next_line.count(']')
            if bracket_depth <= 0:
                content = buffer[:buffer.rfind(']')]
                values = []
                number_pattern = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan"
                for token in re.findall(number_pattern, content, flags=re.IGNORECASE):
                    try:
                        values.append(to_float(token))
                    except ValueError:
                        continue
                stats[key] = values
            i += 1
            continue

        m_val = re.match(r"\s*- \*\*(.+?):\*\*\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|inf|-inf|nan)(?!\s*\[)", line, re.IGNORECASE)
        if m_val:
            key = m_val.group(1)
            try:
                stats[key] = to_float(m_val.group(2))
            except ValueError:
                pass
        i += 1

    return stats

def process_tde_statistics(base_dir):
    """
    自动处理所有CounterFactual实验目录下TDE统计，汇总并写入tde_summary.md。
    """

    def mean_of_lists(list_of_lists):
        arr = np.array(list_of_lists)
        return np.mean(arr, axis=0).tolist()

    def std_of_lists(list_of_lists):
        arr = np.array(list_of_lists)
        return np.std(arr, axis=0).tolist()

    def mean_of_scalars(list_of_scalars):
        arr = np.array(list_of_scalars)
        return float(np.mean(arr))

    def std_of_scalars(list_of_scalars):
        arr = np.array(list_of_scalars)
        return float(np.std(arr))

    def median_of_scalars(list_of_scalars):
        arr = np.array(list_of_scalars)
        return float(np.median(arr))

    if os.path.exists(base_dir):
        for exp_dir in os.listdir(base_dir):
            exp_path = os.path.join(base_dir, exp_dir)
            if not os.path.isdir(exp_path):
                continue
            fold_stats = []
            fold_contents = []
            for fold in range(5):
                md_path = os.path.join(exp_path, f'fold{fold}/eval/tde_stats_epoch_0.md')
                if os.path.exists(md_path):
                    with open(md_path, 'r') as f:
                        lines = f.readlines()
                    fold_contents.append(f'# Fold {fold}\n' + ''.join(lines) + '\n')
                    stats = parse_tde_stats(lines)
                    fold_stats.append(stats)
                else:
                    fold_contents.append(f'# Fold {fold}\nNot found: {md_path}\n')
            # 汇总所有fold内容
            summary_path = os.path.join(exp_path, 'tde_summary.md')
            with open(summary_path, 'w') as f:
                f.write(f'# TDE Statistics Summary for {exp_dir}\n\n')
                for content in fold_contents:
                    f.write(content + '\n')
                # 计算均值、std、median
                if fold_stats:
                    all_keys = set()
                    for d in fold_stats:
                        all_keys.update(d.keys())
                    f.write('\n# 5-Fold Mean Statistics\n')
                    for key in sorted(all_keys):
                        vals = [d[key] for d in fold_stats if key in d]
                        if not vals:
                            continue
                        if isinstance(vals[0], list):
                            mean_val = mean_of_lists(vals)
                            std_val = std_of_lists(vals)
                            mean_std_str = '[' + ' '.join(f'{m:.6f}+{s:.6f}' for m, s in zip(mean_val, std_val)) + ']'
                            f.write(f'- **{key} (mean+std):** {mean_std_str}\n')
                        else:
                            mean_val = mean_of_scalars(vals)
                            std_val = std_of_scalars(vals)
                            if key == 'Overall OR Median':
                                # OR Median 本身是中位数统计量，5折汇总用 median of medians
                                median_val = median_of_scalars(vals)
                                f.write(f'- **{key}:** {median_val:.8f}\n')
                            else:
                                f.write(f'- **{key} (mean+std):** {mean_val:.8f}+{std_val:.8f}\n')

def compare_with_q2l_and_generate_heatmaps(cf_subdir, dataset_name, output_dir=None):
    """
    对比指定 Counterfactual 子目录与 Q2L baseline，生成热力图。
    
    参数：
        cf_subdir: Counterfactual 子目录的完整路径 (e.g., 'work_dirs_MICCAI2026_new/ODIR/Counterfactual_Top_mask')
        dataset_name: 数据集名称 (e.g., 'ODIR', 'Endo', 'FFA')
        output_dir: 热力图输出目录（如果为 None，则与 cf_subdir 同级）
    """
    
    import seaborn as sns
    
    # 验证 cf_subdir 存在
    if not os.path.exists(cf_subdir):
        print(f"Error: Counterfactual directory not found: {cf_subdir}")
        return
    
    # 确定 Q2L 路径
    parent_dir = os.path.dirname(cf_subdir)  # 获取 {dataset_name} 目录
    q2l_path = os.path.join(parent_dir, 'Q2L')
    
    if not os.path.exists(q2l_path):
        print(f"Error: Q2L directory not found: {q2l_path}")
        return
    
    # 确定输出目录
    if output_dir is None:
        # 与 cf_subdir 同级生成 heatmap_comparison 目录
        cf_method_name = os.path.basename(cf_subdir)
        output_dir = os.path.join(parent_dir, f'heatmap_comparison_{cf_method_name}')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 读取 Q2L baseline 结果
    print(f"Reading Q2L baseline from: {q2l_path}")
    q2l_results = {}
    
    for folder in os.listdir(q2l_path):
        folder_path = os.path.join(q2l_path, folder)
        if not os.path.isdir(folder_path):
            continue
        
        # 提取 mAUC
        mauc_mean, mauc_std = extract_mean_mauc(os.path.join(folder_path, '5fold_results.md'))
        
        # 提取 TDE 指标
        tde_stats = {}
        tde_path = os.path.join(folder_path, 'tde_summary.md')
        for metric in ['Overall OR Median', 'Overall TDE Negative', 'Overall TDE Positive']:
            mean_val, std_val = extract_tde_value(tde_path, metric)
            tde_stats[metric] = (mean_val, std_val)
        
        q2l_results[folder] = {
            'mAUC': (mauc_mean, mauc_std),
            **tde_stats
        }
    
    if not q2l_results:
        print(f"Error: No Q2L baseline found in {q2l_path}")
        return
    
    # 获取 Q2L 的参数（通常只有一个）
    q2l_key = list(q2l_results.keys())[0]
    print(f"Using Q2L baseline: {q2l_key}\n")
    
    # 2. 读取 Counterfactual 结果（从 cf_subdir）
    print(f"Reading Counterfactual results from: {cf_subdir}")
    cf_results = {}
    
    for folder in os.listdir(cf_subdir):
        folder_path = os.path.join(cf_subdir, folder)
        if not os.path.isdir(folder_path):
            continue
        
        # 提取 mAUC
        mauc_mean, mauc_std = extract_mean_mauc(os.path.join(folder_path, '5fold_results.md'))
        
        # 提取 TDE 指标
        tde_stats = {}
        tde_path = os.path.join(folder_path, 'tde_summary.md')
        for metric in ['Overall OR Median', 'Overall TDE Negative', 'Overall TDE Positive']:
            mean_val, std_val = extract_tde_value(tde_path, metric)
            tde_stats[metric] = (mean_val, std_val)
        
        cf_results[folder] = {
            'mAUC': (mauc_mean, mauc_std),
            **tde_stats
        }
    
    if not cf_results:
        print(f"Error: No results found in {cf_subdir}")
        return
    
    print(f"Found {len(cf_results)} parameter combinations\n")
    
    # 3. 提取参数和指标，生成对比表格
    metrics = ['mAUC', 'Overall OR Median', 'Overall TDE Negative', 'Overall TDE Positive']
    
    for metric in metrics:
        print(f"{'='*80}")
        print(f"Generating heatmap for: {metric}")
        print(f"{'='*80}")
        
        # 获取 Q2L 值
        q2l_value = q2l_results[q2l_key].get(metric)
        if q2l_value and q2l_value[0] is not None:
            q2l_val = float(q2l_value[0])
        else:
            print(f"Warning: Q2L {metric} not found")
            continue
        
        # 提取所有 cf 和 margin 参数
        cf_values = set()
        margin_values = set()
        all_params = {}  # (cf, margin) -> (cf_val, diff_value)
        
        for param_key in sorted(cf_results.keys()):
            cf, margin, lambda_val, lpos, lneg = extract_param_from_folder(param_key)
            if cf is not None and margin is not None:
                cf_values.add(cf)
                margin_values.add(margin)
                
                val = cf_results[param_key].get(metric)
                if val and val[0] is not None:
                    cf_val = float(val[0])
                    # 对于 mAUC，乘以 100
                    if metric == 'mAUC':
                        cf_val_display = cf_val * 100
                        q2l_val_display = q2l_val * 100
                    else:
                        cf_val_display = cf_val
                        q2l_val_display = q2l_val
                    
                    # 计算差值（绝对差）
                    diff_value = cf_val_display - q2l_val_display
                    all_params[(cf, margin)] = (cf_val_display, diff_value)
        
        if not all_params:
            print(f"No data available for {metric}\n")
            continue
        
        # 4. 创建热力图数据结构（行=cf，列=margin）
        cf_list = sorted(cf_values)
        margin_list = sorted(margin_values)
        
        heatmap_data_diff = np.zeros((len(cf_list), len(margin_list)))
        heatmap_data_value = np.zeros((len(cf_list), len(margin_list)))
        heatmap_text = [['' for _ in range(len(margin_list))] for _ in range(len(cf_list))]
        
        for i, cf in enumerate(cf_list):
            for j, margin in enumerate(margin_list):
                if (cf, margin) in all_params:
                    cf_val_display, diff_value = all_params[(cf, margin)]
                    heatmap_data_value[i, j] = cf_val_display
                    heatmap_data_diff[i, j] = diff_value
                    # 创建标注文本：只显示差值
                    heatmap_text[i][j] = f'{diff_value:+.2f}'
                else:
                    heatmap_text[i][j] = 'N/A'
        
        # 5. 绘制热力图
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 对于 mAUC，显示的 Q2L 值也要乘以 100
        if metric == 'mAUC':
            q2l_val_display = q2l_val * 100
        else:
            q2l_val_display = q2l_val
        
        # 左图：相对改进差值
        ax1 = axes[0]
        sns.heatmap(heatmap_data_diff,
                    xticklabels=[f'{m:.1f}' for m in margin_list],
                    yticklabels=[f'{c:.1f}' for c in cf_list],
                    annot=np.array(heatmap_text),
                    fmt='',
                    cmap='RdYlGn',
                    center=0,
                    cbar_kws={'label': 'Difference'},
                    ax=ax1,
                    linewidths=1,
                    linecolor='gray')
        ax1.set_title(f'{metric} - Difference vs Q2L\n(Q2L={q2l_val_display:.2f})', fontsize=12, fontweight='bold')
        ax1.set_xlabel('margin', fontsize=11)
        ax1.set_ylabel('cf', fontsize=11)
        
        # 右图：绝对值
        ax2 = axes[1]
        sns.heatmap(heatmap_data_value,
                    xticklabels=[f'{m:.1f}' for m in margin_list],
                    yticklabels=[f'{c:.1f}' for c in cf_list],
                    annot=heatmap_data_value,
                    fmt='.2f',
                    cmap='viridis',
                    cbar_kws={'label': metric},
                    ax=ax2,
                    linewidths=1,
                    linecolor='gray')
        ax2.set_title(f'{metric} - Absolute Values', fontsize=12, fontweight='bold')
        ax2.set_xlabel('margin', fontsize=11)
        ax2.set_ylabel('cf', fontsize=11)
        
        plt.tight_layout()
        
        # 6. 保存热力图
        sanitized_metric = metric.replace('/', '_').replace(' ', '_')
        cf_method_name = os.path.basename(cf_subdir)
        heatmap_path = os.path.join(output_dir, f'{dataset_name}_{cf_method_name}_{sanitized_metric}_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        print(f"✓ Heatmap saved: {heatmap_path}")
        plt.close()
        
        # 7. 生成对比表格 (Markdown)
        md_lines = [f'\n## {metric} - {cf_method_name} vs Q2L\n']
        md_lines.append(f'**Q2L Baseline Value:** {q2l_val_display:.2f}\n\n')
        md_lines.append('| cf \\ margin | ' + ' | '.join(f'{m:.1f}' for m in margin_list) + ' |\n')
        md_lines.append('|' + '---|' * (len(margin_list) + 1))
        
        for cf in cf_list:
            row = [f'{cf:.1f}']
            for margin in margin_list:
                if (cf, margin) in all_params:
                    cf_val_display, diff_value = all_params[(cf, margin)]
                    # 只显示差值
                    row.append(f'{diff_value:+.2f}')
                else:
                    row.append('N/A')
            md_lines.append('| ' + ' | '.join(row) + ' |\n')
        
        # 保存对比表格
        table_md_path = os.path.join(output_dir, f'{dataset_name}_{cf_method_name}_{sanitized_metric}_comparison.md')
        with open(table_md_path, 'w') as f:
            f.write('\n'.join(md_lines))
        print(f"✓ Comparison table saved: {table_md_path}\n")
    
    print(f"{'='*80}")
    print(f"All heatmaps and comparison tables saved to: {output_dir}")
    print(f"{'='*80}\n")


def main():
    # 可根据需要自定义路径和数据集
    dir_path = 'work_dirs/work_dirs_Prototype_TDE_0602'  # 根目录
    dataset = 'FFA'  # 只需指定一次数据集名
    subdir = f'{dataset}/Lambda_Ablation_margin0.5_new'  # 子目录
    process_classification_metrics(os.path.join(dir_path, subdir))
    process_tde_statistics(os.path.join(dir_path, subdir))

    print(f"\n================= {dataset} 参数表格统计 =================")
    generate_cf_margin_tables(os.path.join(dir_path, subdir), dataset, output_md_path=os.path.join(dir_path, dataset,f'{dataset}_Lambda_Ablation_margin0.5_new.md'))
    
    # 新增：生成对比 Q2L 的热力图（只对比指定的 subdir 与 Q2L）
    # print(f"\n================= 生成 {dataset} 的 {os.path.basename(subdir)} vs Q2L 对比热力图 =================")
    # compare_with_q2l_and_generate_heatmaps(os.path.join(dir_path, subdir), dataset)

if __name__ == "__main__":
    main()
