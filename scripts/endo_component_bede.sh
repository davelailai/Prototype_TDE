#!/bin/bash
#SBATCH --account=bdliv04
#SBATCH --time=2-00:00:00
#SBATCH --job-name=Endo_component
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-6
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=Jianyang.Xie@liverpool.ac.uk
#SBATCH --exclude=gpu025,gpu026,gpu028
#SBATCH --output=logs/Endo_component_%A_%a.out
#SBATCH --error=logs/Endo_component_%A_%a.err

set -u
export LANG=C
export LC_ALL=C

module load gcc/12.2 || exit 1
module load cuda/12.4.1 || exit 1
CONDA_BASE=${CONDA_BASE:-/nobackup/projects/bdliv04/jxie/miniconda}
source "$CONDA_BASE/etc/profile.d/conda.sh" || exit 1
conda activate mmcls || exit 1

PROJECT_ROOT=${PROJECT_ROOT:-/nobackup/projects/bdliv04/jxie/Prototype_TDE}
cd "$PROJECT_ROOT" || exit 1
mkdir -p logs

CONFIG_FILE='configs/Endo/Endo_CounterFactual_TDE.py'
BASE_WORK_DIR=${ENDO_COMPONENT_ROOT:-work_dirs/Endo_patient_disjoint_direct/Component_Ablation}
ANN_DIR=${ENDO_ANN_DIR:-data/Endo/Endo/patient_disjoint_balanced_seed42}
IMAGE_ROOT=${IMAGE_ROOT:-/nobackup/projects/bdliv04/jxie/SCIF/data}

# all_loss is the selected margin=0.5, lambda_pos=0.8, lambda_neg=1.0,
# orthogonality_weight=0.1 experiment. Reuse it without retraining.
mkdir -p "$BASE_WORK_DIR"
if [[ ! -e "$BASE_WORK_DIR/all_loss" ]]; then
    ln -s ../Lambda_Ablation_margin0.5/lpos0.8_lneg1.0 "$BASE_WORK_DIR/all_loss"
fi

variants=(baseline only_lpos only_lneg only_ortho no_lpos no_lneg no_ortho)
lambda_pos=(0.0 0.8 0.0 0.0 0.0 0.8 0.8)
lambda_neg=(0.0 0.0 1.0 0.0 1.0 0.0 1.0)
ortho=(0.0 0.0 0.0 0.1 0.1 0.1 0.0)

task=${SLURM_ARRAY_TASK_ID:?Submit this script with sbatch as an array job}
if (( task < 0 || task >= ${#variants[@]} )); then
    echo "Invalid task index $task; expected 0-6" >&2
    exit 2
fi

variant=${variants[$task]}
lpos=${lambda_pos[$task]}
lneg=${lambda_neg[$task]}
orthogonality_weight=${ortho[$task]}
intervention_margin=0.5

echo "========================================================"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Variant: $variant"
echo "lambda_pos=$lpos lambda_neg=$lneg ortho=$orthogonality_weight margin=$intervention_margin"
echo "Patient-independent annotation directory: $ANN_DIR"
echo "Output: $BASE_WORK_DIR/$variant"
echo "========================================================"

failed=0
for seed in 0 1 2 3 4; do
    work_dir="$BASE_WORK_DIR/$variant/fold$seed"
    if [[ -e "$work_dir" ]]; then
        echo "Refusing to overwrite existing run: $work_dir" >&2
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
        model.head.task_heads.classification.lambda_interv_pos="$lpos" \
        model.head.task_heads.classification.lambda_interv_neg="$lneg" \
        model.head.task_heads.classification.orthogonality_weight="$orthogonality_weight" \
        model.head.task_heads.classification.prototype_init=query_mlp \
        model.head.task_heads.classification.intervention_mode=proj_gate \
        model.head.task_heads.classification.train_with_TDE=False \
        model.head.task_heads.classification.prototype_contextual=False \
        train_dataloader.dataset.ann_file="$ANN_DIR/train_list.pkl" \
        val_dataloader.dataset.ann_file="$ANN_DIR/val_list.pkl" \
        test_dataloader.dataset.ann_file="$ANN_DIR/test_list.pkl" \
        train_dataloader.dataset.data_prefix.img_path="$IMAGE_ROOT" \
        val_dataloader.dataset.data_prefix.img_path="$IMAGE_ROOT" \
        test_dataloader.dataset.data_prefix.img_path="$IMAGE_ROOT" \
        default_hooks.checkpoint.save_best=multi-class/mAUC_multiclass \
        default_hooks.checkpoint.rule=greater; then
        echo "SUCCESS variant=$variant seed=$seed"
    else
        failed=1
        echo "FAILED variant=$variant seed=$seed" >&2
        echo "variant=$variant seed=$seed" >> \
            "$BASE_WORK_DIR/failed_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
    fi
done

echo "Finished: $(date)"
exit "$failed"
