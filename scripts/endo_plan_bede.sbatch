#!/usr/bin/env bash
#SBATCH --account=bdliv04
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=Endo_plan
#SBATCH --array=0-41%2
#SBATCH --output=logs/Endo_plan_%A_%a.out
#SBATCH --error=logs/Endo_plan_%A_%a.err

set -euo pipefail

# Submit from the project root with the existing training environment active.
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
cd "$PROJECT_ROOT"
PLAN=$(realpath "${1:?Usage: sbatch scripts/endo_plan_bede.sbatch PLAN.json}")
MODE=${2:-run}
PYTHON_BIN=${PYTHON_BIN:-python}
case "$MODE" in
    run) TASK_INDEX=${SLURM_ARRAY_TASK_ID:?Use sbatch, or pass dry-run as the second argument} ;;
    dry-run|count) TASK_INDEX=${SLURM_ARRAY_TASK_ID:-0} ;;
    *) echo "Unsupported mode: $MODE" >&2; exit 1 ;;
esac

# Read the existing plan; keep all seeds of one setting in the same array job.
TASK_IDS=$("$PYTHON_BIN" - "$PLAN" "$MODE" "$TASK_INDEX" <<'PY'
import json
from pathlib import Path
import sys

plan_path, mode, index = sys.argv[1:]
plan = json.loads(Path(plan_path).read_text())
groups = {}
for position, task in enumerate(plan['tasks']):
    if task['index'] != position:
        raise ValueError('Plan task indices are inconsistent.')
    key = (task['code_root'], str(Path(task['work_dir']).parent))
    groups.setdefault(key, []).append(task)
if not groups:
    raise ValueError('Empty plan.')
for group in groups.values():
    if sorted(t['seed'] for t in group) != sorted(plan['seeds']):
        raise ValueError('A setting has missing or duplicate seeds.')
if mode == 'count':
    print(len(groups))
else:
    index = int(index)
    if not 0 <= index < len(groups):
        raise ValueError(f'Array index {index} outside 0-{len(groups)-1}.')
    group = list(groups.values())[index]
    for task in sorted(group, key=lambda t: t['seed']):
        print(task['index'])
PY
)

if [[ "$MODE" == count ]]; then
    echo "$TASK_IDS"
    exit 0
fi
mapfile -t IDS <<< "$TASK_IDS"

echo "========================================================"
echo "Started: $(date)"
date_start=$(date +%s)
echo "Node: $(hostname)"
echo "Job: ${SLURM_ARRAY_JOB_ID:-dry-run}; setting: $TASK_INDEX"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Plan: $PLAN"
echo "Sequential runs for this setting: ${#IDS[@]}"
echo "========================================================"

failed=0
for task_id in "${IDS[@]}"; do
    echo "Starting plan task $task_id at $(date)"
    if [[ "$MODE" == dry-run ]]; then
        "$PYTHON_BIN" tools/Analysis/run_endo_task.py \
            --plan "$PLAN" --task-id "$task_id" --dry-run
    elif srun --ntasks=1 "$PYTHON_BIN" -u tools/Analysis/run_endo_task.py \
        --plan "$PLAN" --task-id "$task_id"; then
        echo "SUCCESS: plan task $task_id"
    else
        failed=1
        echo "FAILED: plan task $task_id; continuing with the next seed"
        printf 'setting=%s task=%s\n' "$TASK_INDEX" "$task_id" >> \
            "$(dirname "$PLAN")/failed_${SLURM_ARRAY_JOB_ID}_${TASK_INDEX}.log"
    fi
done

echo "Finished: $(date)"
echo "Elapsed seconds: $(($(date +%s) - date_start))"
exit "$failed"
