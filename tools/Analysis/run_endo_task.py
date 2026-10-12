"""Run one prepared Endo task, with a work-directory lock and completion marker."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--task-id', type=int, required=True)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    if not 0 <= args.task_id < len(plan['tasks']):
        raise ValueError('Task index out of range.')
    task = plan['tasks'][args.task_id]
    work, config, code = map(Path, (task['work_dir'], task['config'], task['code_root']))
    for filename, checksum in plan['annotation_sha256'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != checksum:
            raise ValueError(f'Annotation changed after plan creation: {filename}')
    train = [sys.executable, '-u', str(code / 'tools/Train.py'), str(config), '--work-dir', str(work)]
    print(json.dumps(task, indent=2), flush=True)
    print('Train:', train, flush=True)
    if args.dry_run:
        return
    with (work / '.task.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Task is already running: {work}')
        if (work / 'DONE.json').exists():
            print('Completed task; skipping.', flush=True)
            return
        existing = list(work.glob('*.pth'))
        if existing:
            raise RuntimeError(f'Incomplete run has checkpoints in {work}; inspect it before restarting.')
        environment = os.environ.copy()
        environment['OMP_NUM_THREADS'] = str(os.environ.get('SLURM_CPUS_PER_TASK', '8'))
        subprocess.run(train, cwd=code, env=environment, check=True)
        checkpoints = list(work.glob('best_multi-class_mAUC_multiclass_epoch*.pth'))
        if len(checkpoints) != 1:
            raise RuntimeError(f'Expected exactly one best validation-AUC checkpoint: {checkpoints}')
        evaluation = work / 'eval'
        evaluation.mkdir(exist_ok=True)
        test = [sys.executable, '-u', str(code / 'tools/Test.py'), str(config), str(checkpoints[0]),
                '--work-dir', str(evaluation), '--out-item', 'pred',
                '--out', str(evaluation / f'pred_results_fold{task["seed"]}.pkl')]
        subprocess.run(test, cwd=code, env=environment, check=True)
        (work / 'DONE.json').write_text(json.dumps({'task': task, 'checkpoint': str(checkpoints[0])}, indent=2))


if __name__ == '__main__':
    main()
