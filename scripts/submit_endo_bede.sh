#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/submit_endo_bede.sh PLAN.json [MAX_PARALLEL=2]"
    echo "Run from your activated training environment; it is inherited by the array."
    exit 1
fi
PLAN=$(realpath "$1")
MAX_PARALLEL=${2:-2}
[[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid concurrency"; exit 1; }
PYTHON_BIN=$(command -v python)
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
COUNT=$("$PYTHON_BIN" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["tasks"]))' "$PLAN")
[[ "$COUNT" -gt 0 ]] || { echo "Empty plan"; exit 1; }
LOG_DIR="$(dirname "$PLAN")/slurm_logs"
mkdir -p "$LOG_DIR"
exec 9>"$(dirname "$PLAN")/.submission.lock"
flock -n 9 || { echo "Another submission is in progress for this plan"; exit 1; }
if [[ -f "$(dirname "$PLAN")/submitted_job_id.txt" ]]; then
    echo "This plan was already submitted. Check submitted_job_id.txt and squeue."
    exit 1
fi
"$PYTHON_BIN" "$PROJECT_ROOT/tools/Analysis/run_endo_task.py" --plan "$PLAN" --task-id 0 --dry-run
JOB_ID=$(sbatch --parsable --account=bdliv04 --partition=gpu --nodes=1 \
    --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=2-00:00:00 \
    --export=ALL --job-name=Endo_patient_all \
    --array="0-$((COUNT-1))%${MAX_PARALLEL}" \
    --output="$LOG_DIR/%A_%a.out" --error="$LOG_DIR/%A_%a.err" \
    "$PROJECT_ROOT/scripts/endo_bede_task.sh" "$PYTHON_BIN" "$PROJECT_ROOT" "$PLAN")
printf '%s\n' "$JOB_ID" > "$(dirname "$PLAN")/submitted_job_id.txt"
echo "Submitted $COUNT tasks, at most $MAX_PARALLEL concurrent: $JOB_ID"
