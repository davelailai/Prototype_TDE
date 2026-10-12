#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN=$1
PROJECT_ROOT=$2
PLAN=$3
cd "$PROJECT_ROOT"
srun --ntasks=1 "$PYTHON_BIN" -u tools/Analysis/run_endo_task.py \
    --plan "$PLAN" --task-id "${SLURM_ARRAY_TASK_ID:?Missing Slurm array task ID}"
