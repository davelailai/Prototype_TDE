#!/bin/bash
#SBATCH --account=bdliv04
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=SOTA
#SBATCH --array=0-14
#SBATCH --mail-type=FAIL

set -euo pipefail
export LC_ALL=C
module load gcc/12.2
module load cuda/12.4.1
# Activate mmcls before sbatch; do not assume source activate exists on nodes.
: "${CONDA_PREFIX:?Activate mmcls before submitting this script}"
PYTHON="${CONDA_PREFIX}/bin/python"
cd "${SLURM_SUBMIT_DIR}"

methods=(MLC-PAT CIPL ProCI MICCAI2024 BankCausal)
datasets=(FFA Endo ODIR)
task=${SLURM_ARRAY_TASK_ID}
method=${methods[$((task / 3))]}
dataset=${datasets[$((task % 3))]}
image_root=${IMAGE_ROOT:-/nobackup/projects/bdliv04/jxie/SCIF/data}
protocol=${SOTA_PROTOCOL:-aligned}
base_work=${SOTA_WORK_ROOT:-work_dirs/SOTA_${protocol}}
status=0
for seed in 0 1 2 3 4; do
  if [ "${method}" = BankCausal ]; then
    work="${base_work}/${dataset}/${method}/seed${seed}"
    if "${PYTHON}" -B tools/SOTA/prepare_prior.py \
      --dataset "${dataset}" --image-root "${image_root}" --seed "${seed}" \
      --protocol "${protocol}" --batch-size "${BATCH_SIZE:-32}" --work-dir "${work}"; then
      if ! "${PYTHON}" -B tools/SOTA/train_bankcausal.py "${work}/resolved.py"; then
        status=1
      fi
    else
      status=1
    fi
    continue
  fi
  if ! "${PYTHON}" -B tools/SOTA/train.py \
    --method "${method}" --dataset "${dataset}" \
    --image-root "${image_root}" --seed "${seed}" \
    --protocol "${protocol}" --batch-size "${BATCH_SIZE:-32}" \
    --work-dir "${base_work}/${dataset}/${method}/seed${seed}"; then
    echo "FAILED ${method} ${dataset} seed=${seed}" >&2
    status=1
  fi
done
exit "${status}"
