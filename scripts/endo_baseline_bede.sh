#!/bin/bash
#SBATCH --account=bdliv04
#SBATCH -t 2-00:00:00
#SBATCH --job-name=Endo_baselines
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=Jianyang.Xie@liverpool.ac.uk
#SBATCH --exclude=gpu025,gpu028,gpu026
#SBATCH --output=logs/Endo_baseline_%A_%a.out
#SBATCH --error=logs/Endo_baseline_%A_%a.err

export LANG=C
export LC_ALL=C
module load gcc/12.2 || exit 1
module load cuda/12.4.1 || exit 1
CONDA_BASE=${CONDA_BASE:-/nobackup/projects/bdliv04/jxie/miniconda}
source "$CONDA_BASE/etc/profile.d/conda.sh" || exit 1
conda activate mmcls || exit 1

PROJECT_ROOT='/nobackup/projects/bdliv04/jxie/Prototype_TDE'
BASELINE_ROOT=${BASELINE_ROOT:-/nobackup/projects/bdliv04/jxie/Multi-disease-diagnosis}
BASE_WORK_DIR="$PROJECT_ROOT/work_dirs/Endo_patient_disjoint_direct/Main_Comparison"
ANN_DIR="$PROJECT_ROOT/data/Endo/Endo/patient_disjoint_balanced_seed42"
IMAGE_ROOT='/nobackup/projects/bdliv04/jxie/SCIF/data'
METHODS=(Base Asy dyGCN MLDecoder Q2L)
CONFIGS=(Endo_baseline.py Endo_AsyLoss.py Endo_GCNhead.py Endo_MLDecoder.py Endo_Q2L.py)

if [[ ${SLURM_ARRAY_TASK_ID} -lt 0 || ${SLURM_ARRAY_TASK_ID} -ge 5 ]]; then
    echo "Invalid baseline array index: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
CONFIG_FILE="$PROJECT_ROOT/configs/Endo/${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

# Use the complete project's registered baseline heads with the new shared configs.
cd "$BASELINE_ROOT" || exit 1
[[ -f tools/Train.py && -f "$CONFIG_FILE" ]] || {
    echo "Missing training entry or baseline config: $BASELINE_ROOT, $CONFIG_FILE" >&2
    exit 1
}
for split in train val test; do
    [[ -f "$ANN_DIR/${split}_list.pkl" ]] || { echo "Missing $split annotation" >&2; exit 1; }
done

echo "========================================================"
echo "Started: $(date); node: $(hostname)"
date_start=$(date +%s)
echo "Method: $METHOD; sequential seeds: 0 1 2 3 4"
echo "Code root: $BASELINE_ROOT"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "========================================================"
mkdir -p "$BASE_WORK_DIR"
failed=0

for seed in 0 1 2 3 4; do
    work_dir="$BASE_WORK_DIR/$METHOD/fold${seed}"
    echo "Running $METHOD seed=$seed; work_dir=$work_dir"
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
        train_dataloader.dataset.ann_file="$ANN_DIR/train_list.pkl" \
        val_dataloader.dataset.ann_file="$ANN_DIR/val_list.pkl" \
        test_dataloader.dataset.ann_file="$ANN_DIR/test_list.pkl" \
        train_dataloader.dataset.data_prefix.img_path="$IMAGE_ROOT" \
        val_dataloader.dataset.data_prefix.img_path="$IMAGE_ROOT" \
        test_dataloader.dataset.data_prefix.img_path="$IMAGE_ROOT" \
        default_hooks.checkpoint.save_best=multi-class/mAUC_multiclass \
        default_hooks.checkpoint.rule=greater; then
        echo "SUCCESS: $METHOD seed=$seed"
    else
        failed=1
        echo "FAILED: $METHOD seed=$seed; continuing with the next seed"
        echo "method=$METHOD seed=$seed" >> \
            "$BASE_WORK_DIR/failed_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
    fi
done

echo "Finished: $(date)"
echo "Elapsed seconds: $(($(date +%s) - date_start))"
exit "$failed"
