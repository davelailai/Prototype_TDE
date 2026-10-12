#!/bin/bash
#SBATCH --account=bdliv04
#SBATCH -t 2-00:00:00
#SBATCH --job-name=Endo_margin_ablation
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-5
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=Jianyang.Xie@liverpool.ac.uk
#SBATCH --exclude=gpu025,gpu028,gpu026
#SBATCH --output=logs/Endo_margin_%A_%a.out
#SBATCH --error=logs/Endo_margin_%A_%a.err

export LANG=C
export LC_ALL=C

module load gcc/12.2 || exit 1
module load cuda/12.4.1 || exit 1
CONDA_BASE=${CONDA_BASE:-/nobackup/projects/bdliv04/jxie/miniconda}
source "$CONDA_BASE/etc/profile.d/conda.sh" || exit 1
conda activate mmcls || exit 1

cd /nobackup/projects/bdliv04/jxie/Prototype_TDE || exit 1

echo "========================================================"
echo "Started: $(date)"
date_start=$(date +%s)
hostname
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "GPU_DEVICE_ORDINAL: ${GPU_DEVICE_ORDINAL:-unset}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "========================================================"

CONFIG_FILE='configs/Endo/Endo_CounterFactual_TDE.py'
BASE_WORK_DIR='work_dirs/Endo_patient_disjoint_direct/Margin_Ablation'
ANN_DIR='data/Endo/Endo/patient_disjoint_balanced_seed42'
IMAGE_ROOT='/nobackup/projects/bdliv04/jxie/SCIF/data'

prototype_init='query_mlp'
orthogonality_weight=0.1
intervention_mode='proj_gate'
prototype_contextual=False
train_with_TDE=False
lambda_interv_pos=1.0
lambda_interv_neg=1.0
intervention_margin_values=(0.1 0.3 0.5 0.7 0.9 1.1)

if [[ ${SLURM_ARRAY_TASK_ID} -lt 0 || ${SLURM_ARRAY_TASK_ID} -ge 6 ]]; then
    echo "Invalid margin array index: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi
intervention_margin=${intervention_margin_values[$SLURM_ARRAY_TASK_ID]}
echo "Margin: $intervention_margin; sequential seeds: 0 1 2 3 4"
mkdir -p "$BASE_WORK_DIR"
failed=0

for seed in 0 1 2 3 4; do
    work_dir="${BASE_WORK_DIR}/margin${intervention_margin}/fold${seed}"
    echo "========================================================"
    echo "Running seed ${seed} with margin=${intervention_margin}"
    echo "Work directory: ${work_dir}"

    if [[ -d "$work_dir" ]]; then
        echo "Existing experiment directory; inspect before rerunning: $work_dir"
        failed=1
        continue
    fi

    if python -u tools/Train.py "$CONFIG_FILE" \
        --work-dir "$work_dir" \
        --cfg-options \
        randomness.seed="$seed" \
        resume=False \
        load_from=None \
        model.head.task_heads.classification.intervention_margin="$intervention_margin" \
        model.head.task_heads.classification.lambda_interv_pos="$lambda_interv_pos" \
        model.head.task_heads.classification.lambda_interv_neg="$lambda_interv_neg" \
        model.head.task_heads.classification.prototype_init="$prototype_init" \
        model.head.task_heads.classification.orthogonality_weight="$orthogonality_weight" \
        model.head.task_heads.classification.intervention_mode="$intervention_mode" \
        model.head.task_heads.classification.train_with_TDE="$train_with_TDE" \
        model.head.task_heads.classification.prototype_contextual="$prototype_contextual" \
        train_dataloader.dataset.ann_file="$ANN_DIR/train_list.pkl" \
        val_dataloader.dataset.ann_file="$ANN_DIR/val_list.pkl" \
        test_dataloader.dataset.ann_file="$ANN_DIR/test_list.pkl" \
        train_dataloader.dataset.data_prefix.img_path="$IMAGE_ROOT" \
        val_dataloader.dataset.data_prefix.img_path="$IMAGE_ROOT" \
        test_dataloader.dataset.data_prefix.img_path="$IMAGE_ROOT" \
        default_hooks.checkpoint.save_best=multi-class/mAUC_multiclass \
        default_hooks.checkpoint.rule=greater; then
        echo "SUCCESS: margin=${intervention_margin} seed=${seed}"
    else
        failed=1
        echo "FAILED: margin=${intervention_margin} seed=${seed}"
        echo "margin=$intervention_margin seed=$seed" >> \
            "$BASE_WORK_DIR/failed_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
    fi
done

echo "Finished: $(date)"
echo "Elapsed seconds: $(($(date +%s) - date_start))"
exit "$failed"
