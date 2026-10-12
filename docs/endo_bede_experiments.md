# Endo patient-independent experiments on Bede

Use your existing activated training environment. The Slurm array inherits it;
the submission scripts do not install packages or change environments.

## Prepare the experiment plan

```bash
cd /nobackup/projects/bdliv04/jxie/Prototype_TDE
git pull origin main

python tools/Analysis/prepare_endo_suite.py \
  --baseline-root /nobackup/projects/bdliv04/jxie/Multi-disease-diagnosis \
  --image-root /nobackup/projects/bdliv04/jxie/SCIF/data \
  --annotation-root data \
  --suite-dir work_dirs/Endo_patient_disjoint_v1
```

The three Endo annotations must already have been restored from the committed
manifests. Preparation verifies their image membership, order, labels, patient
disjointness, and image existence against the manifest. Both absolute-image
and relative-image annotations are supported. It does not start training.

The main baselines run in the existing full `Multi-disease-diagnosis` project,
using its `Endo_baseline.py`, `Endo_AsyLoss.py`, `Endo_GCNhead.py`,
`Endo_MLDecoder.py` and `Endo_Q2L.py` implementations. Q2L-TDE variants run in
`Prototype_TDE`. All generated configs replace data loaders, preprocessing,
backbone/pretraining, optimizer, schedule and evaluation with common settings.
Those five main baseline implementation files must be present on Bede.

The TDE geometry follows the previous Endo ablation: `query_mlp` prototypes,
non-contextual prototypes, projection-gate removal, original prediction at
inference, 100 epochs, and SGD with learning rate 0.005. The all-loss setting
uses positive/negative weights 1/1, purity weight 0.1 and margin 0.5.

| Experiment | Settings | Repetitions |
| --- | ---: | ---: |
| Main comparison | Base, Asy, dyGCN, MLDecoder, Q2L, Q2L-TDE | 5 each |
| Component | 8 combinations | 5 each |
| Margin | 0.1, 0.3, 0.5, 0.7, 0.9, 1.1 | 5 each |
| Lambda | 0.2, 0.4, 0.6, 0.8, 1.0 for each weight | 5 each |

The full-model setting shared by these groups is trained once per seed. The
default suite has 42 unique settings and 210 training+test tasks. `plan.json`
records the aliases so repeated table entries use the same results.

The component `baseline` reproduces the zero-loss query geometry control; it
is distinct from the original Q2L implementation in the main comparison.
Review-requested additional methods are not silently substituted for these
five existing baselines and need their own validated configurations.

Seeds 0-4 are five repetitions on the same fixed patient split, not five-fold
cross-validation. Validation AUC from original predictions selects the best
checkpoint for every model. Test predictions are evaluated only after training.

## Submit

Before submitting, inspect existing Endo jobs to avoid duplicating a suite
already launched outside this tool:

```bash
squeue -u "$USER"
bash scripts/submit_endo_bede.sh \
  work_dirs/Endo_patient_disjoint_v1/plan.json 2
```

Each task requests one V100 GPU, eight CPUs, 32 GB RAM and up to two days in
partition `gpu`, account `bdliv04`. The final argument limits the whole array
to two concurrent tasks. Adjust it to your allocation.

The job ID is saved to `submitted_job_id.txt`. Logs go to `slurm_logs/`;
individual configs/checkpoints go to the task work directory. Evaluation saves
`eval/pred_results_fold<seed>.pkl` and successful tasks receive `DONE.json`.
Duplicate submissions of the same plan and concurrent execution of the same
work directory are blocked. Existing incomplete checkpoint directories require
manual inspection before restarting; old training is not resumed automatically.

To prepare only selected experiment groups, add for example
`--groups main component`. To inspect a task without running it:

```bash
python tools/Analysis/run_endo_task.py \
  --plan work_dirs/Endo_patient_disjoint_v1/plan.json --task-id 0 --dry-run
```

Retain the plan and source code versions with the results. No old image-level
Endo work directories or checkpoints are overwritten.
