#!/bin/bash
#SBATCH --account=bdliv04
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=CIPL_shared
#SBATCH --array=0-2
#SBATCH --mail-type=FAIL

set -euo pipefail
export LC_ALL=C
module load gcc/12.2
module load cuda/12.4.1
: "${CONDA_PREFIX:?Activate mmcls before submitting this script}"
cd "${SLURM_SUBMIT_DIR}"
PYTHON="${CONDA_PREFIX}/bin/python"
datasets=(FFA Endo ODIR)
task=${SLURM_ARRAY_TASK_ID:?Submit as an array job}
if (( task < 0 || task >= ${#datasets[@]} )); then
  echo "Invalid array task ${task}; expected 0-2" >&2
  exit 2
fi
dataset=${datasets[$task]}
lr=${SOTA_LR:-0.005}
base=${SOTA_WORK_ROOT:-work_dirs/CIPL_official_shared_lr${lr}_${SLURM_ARRAY_JOB_ID}}
status=0
echo "CIPL dataset=${dataset} base_lr=${lr} job=${SLURM_ARRAY_JOB_ID}"
echo "Recipe: 15 warmup + 26 full epochs, push at 35/40, 6 refinement epochs per push"
for seed in 0 1 2 3 4; do
  work="${base}/${dataset}/CIPL/seed${seed}"
  echo "Running seed=${seed} output=${work}"
  if ! "${PYTHON}" -B tools/SOTA/train.py \
    --method CIPL --dataset "${dataset}" --protocol aligned \
    --reference-root "${SOTA_REFERENCE_ROOT:-${SLURM_SUBMIT_DIR}}" \
    --source-root "${SOTA_SOURCE_ROOT:-third_party/sota}" \
    --annotation-root "${ANNOTATION_ROOT:-data}" \
    --image-root "${IMAGE_ROOT:-/nobackup/projects/bdliv04/jxie/SCIF/data}" \
    --lr "${lr}" --batch-size "${BATCH_SIZE:-32}" --workers "${WORKERS:-4}" \
    --seed "${seed}" --work-dir "${work}"; then
    echo "FAILED ${dataset} CIPL seed=${seed}" >&2
    status=1
  fi
done
exit "${status}"
