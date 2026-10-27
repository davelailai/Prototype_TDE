# Additional Comparison Methods

These are **controlled medical-dataset adaptations**, not claims to reproduce
the original papers' benchmark numbers. All methods use the existing dataset
annotations, ResNet-50, the project's SwAV initialization, and 448-pixel inputs.
The source revisions are pinned in `sources.json`. No training results have been
generated on the full medical datasets yet.

## Aligned Comparison Protocol (Default)

### Uniform Base Learning Rate Runs

`scripts/sota_bede.sh` explicitly passes `--lr 0.005` to all five methods.
Its default output is a separate `work_dirs/SOTA_aligned_v2_lr0.005` tree;
existing results are not modified. For controlled MICCAI2024, MLC-PAT and
BankCausal runs, the explicit LR must match the reference config. Change the
shared reference protocol to change this LR, not only one competing method.
CIPL's backbone, joint-stage groups and last-layer refinement all use this base
value. CIPL and ProCI retain their method-specific optimizers/schedules.
MICCAI2024, MLC-PAT and BankCausal use the exact reference SGD/schedulers in
new aligned training. Historical/native PAT runs retain OneCycle.
For CIPL and ProCI it is larger than the previous method-specific default, and
equal numeric learning rates do not guarantee better results or equal tuning.

New aligned MICCAI2024, MLC-PAT and BankCausal entrypoints use reference
optimization by default. CIPL uses reference base LR and its own stages.
Other direct Python entrypoints retain their previous defaults unless `--lr` is supplied.
For example, use `train.py ... --lr 0.005` or
`prepare_prior.py ... --lr 0.005`. The resolved rates are recorded in run settings
and alignment metadata. Old checkpoint evaluation uses its saved settings,
including the legacy CIPL refinement rate when absent from old metadata.

For a fresh server submission (activate the environment first):

```bash
ROOT="work_dirs/SOTA_aligned_lr0.005_$(date +%Y%m%d_%H%M%S)"
sbatch --array=0-14 \
  --export=ALL,SOTA_LR=0.005,SOTA_PROTOCOL=aligned,SOTA_WORK_ROOT="$ROOT" \
  scripts/sota_bede.sh
```

### Corrected CIPL Recipe

New CIPL training records `settings.cipl_recipe = official_mechanisms_v2`.
This restores upstream shared-positive-class sampling (including the normal
class), single-image warmup, lazy teacher initialization at the first full
training forward, full losses during last-layer refinement, a persistent
refinement Adam optimizer, and greedy projection with globally distinct source
images. Both student and teacher views receive independent photometric changes
while sharing spatial geometry. Projection provenance and individual loss terms
are recorded for diagnosis. Projection only searches training images.

The controlled differences remain: the project's ResNet-50/SwAV backbone,
patient-independent annotations and aligned 448-pixel preprocessing, a supported
bottleneck addon (upstream's regular addon hardcodes 256 input channels), and
classification over all disease classes (the upstream full-training call has a
class-count off-by-one). Model selection uses validation mAUC, never test AUC.
This is an upstream-mechanism adaptation, not a bitwise reproduction of published
numbers or the original DenseNet/512-pixel recipe.

Aligned CIPL training defaults to the reference base LR when `--lr` is absent;
the dedicated Bede script explicitly uses 0.005 for backbone, addon, prototypes
and last-layer refinement. It retains Adam, LR halving at epochs 10/20/25/30/35,
15 warmup epochs, 41 main epochs, push at 35/40 and six refinement epochs per
push (53 optimizer dataset passes total). These schedules are not the Q2L
schedule, and equal base LR does not imply equal optimization or compute.
Do not pool the corrected recipe with earlier adapter runs.

After synchronizing the changed files to the server, activate mmcls and submit:

```bash
sbatch --export=ALL,SOTA_LR=0.005 scripts/cipl_official_bede.sh
```

The array runs FFA, Endo and ODIR, each with seeds 0-4, in a fresh job-specific
output tree. `--array=0`, `--array=1` or `--array=2` selects one dataset.
Existing checkpoints and results are never overwritten. CPU verification:

```bash
python -B tools/SOTA/test_cipl_pairing.py
python -B tools/SOTA/test_adapters.py
python -B tools/SOTA/test_cipl_training_smoke.py
```

Set `SOTA_REFERENCE_ROOT` to the ResNet-50 reference checkout when running these
tests from a separate checkout with a different active backbone config.

### Q2L-Controlled Optimization and PAT Memory

New aligned MICCAI2024, MLC-PAT and BankCausal training records
`optimization_recipe=q2l_sgd_v1`. Optimizer parameters and epoch schedulers are
read directly from the reference config, including SGD momentum/weight decay,
10-epoch LinearLR warmup and 90-epoch cosine in the current reference. Standalone
training uses MMEngine's scheduler classes and steps them once after each train
epoch, matching ParamSchedulerHook; PAT's old AdamW/OneCycle and MICCAI2024's
handwritten cosine are not used in this recipe. Epochs default to reference 100.
EMA policies, model architecture, auxiliary objectives and method-specific
intervention stages are retained; this is not identical complete training or
equal compute. Native and historical checkpoint evaluation remain supported.

PAT keeps its full-image-plus-four-patch forward (batch 32 is 160 backbone
images). New training enables ResNet bottleneck activation checkpointing. The
backward wrapper restores BN running statistics/counters to their post-original-
forward values after recomputation, avoiding a second BN update. It does not
split the physical batch or introduce AMP/gradient accumulation. This costs
additional computation. Tests check gradients and buffers, including an actual
CPU ResNet, but GPU memory success depends on device/runtime and must be checked:

```bash
python -B tools/SOTA/check_pat_memory.py --reference-root "$PWD" --batch-size 32
```

If the device still runs out of memory, use `BATCH_SIZE=8` for a new declared
small-batch comparison and rerun Q2L/Q2L-TDE at the same physical batch. Do not
claim that a smaller PAT-only batch is identical to the batch-32 protocol.

After syncing the code to Bede, only these three methods (3 datasets, 5 seeds):

```bash
ROOT="work_dirs/SOTA_q2lopt_$(date +%Y%m%d_%H%M%S)"
sbatch --array=0-2,9-14 \
  --export=ALL,SOTA_LR=0.005,SOTA_PROTOCOL=aligned,SOTA_WORK_ROOT="$ROOT" \
  scripts/sota_bede.sh
```

Unit checks: `test_shared_optimization.py` checks all 100 LR steps against
MMEngine-built reference schedulers and validates checkpointing gradient/BN
parity. `test_training_smoke.py` exercises train/validation/test/checkpoint reload.

`--protocol aligned` resolves the actual per-dataset Q2L-TDE config in this
project, rather than assuming that similarly named loss functions are equivalent.
Use `--reference-root /path/to/Prototype-TDE` or `SOTA_REFERENCE_ROOT` when running
the tools from a different checkout. Unsupported reference backbones fail early;
they are never silently replaced with ResNet-50. The separate older
Multi-disease-diagnosis checkout may have active Swin/ConvNeXt configs.

The aligned protocol uses:

- The reference ResNet-50 architecture, initialization, normalization, and MM image
  operations. Required multi-view/pair/patch construction remains method-specific.
- The project's actual `MILAsymmetricLoss` implementation for PAT, ProCI,
  MICCAI2024 and BankCausal classification branches. In the current reference this
  is gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8, loss_weight=1, with focal
  weights differentiated and loss **summed over classes, divided by batch size**.
  This is neither a full-element mean nor the previous detached-focal sum.
- Exact value AND gradient parity tests against the reference loss. A pretrained
  initialization audit checks required tensor values, not just checkpoint URLs.
- Reference batch size (currently 32) by default. A manual override is recorded
  as a mismatch; gradient accumulation would not make BatchNorm or prototype
  selection equivalent, so no such equivalence is claimed.
- Reference 100 epochs for PAT, ProCI, MICCAI2024 and BankCausal. CIPL retains its
  method-specific 41 joint epochs plus 12 last-layer refinement epochs (53
  optimizer dataset passes); this exception is explicit in the audit report.
- Reference optimizer/schedules for MICCAI2024, MLC-PAT and BankCausal in new
  aligned runs; method-specific optimization for CIPL/ProCI. EMA rules and
  auxiliary objectives remain method-specific and are NOT claimed identical.

For ProCI, the native auxiliary sum is also divided by batch size, keeping its
coefficient convention when the main classification term changes from a sum to
a batch average. Its auxiliary ASL retains the upstream detached-focal behavior;
only its main classification branch is replaced by the reference loss. CIPL's
internal disease/normal paired CE is not replaced with ASL.

The common crop/photometric operations are imported from the reference pipeline;
paired/patch methods apply geometry and photometric steps separately as required
by their view construction. This does not promise bitwise-identical augmentation
draws or identical computational cost across methods.

`--protocol native` retains the preceding adaptation's loss and epoch choices;
it does not mean a complete reproduction of the original paper. Do not pool
native and aligned runs into a single result table. Checkpoints record their
protocol and evaluation refuses a mismatched protocol flag.

Preview settings without training:

```bash
python -B tools/SOTA/audit_protocol.py \
  --output work_dirs/SOTA_aligned/protocol_audit.json
```

Each standalone `run.json`, and each BankCausal `resolved.py`, records the
reference config, resolved loss parameters, exceptions, batch/epoch agreement,
and training pass counts. `initialization_audit` is included in standalone
metadata; BankCausal writes `initialization_audit.json` before training.

## Methods and Adaptation Boundaries

| CLI name | Implementation | Retained mechanisms | Explicit changes |
| --- | --- | --- | --- |
| `MLC-PAT` | PAT-T cross-attention, implemented from the upstream tensor operations | Full image plus 2x2 patches, three classifiers, patch-logit softmax attention, full/patch ASL, averaged-logit inference, EMA | Shared ResNet-50/loss/image operations; reference SGD/schedulers and activation checkpointing in new aligned runs (AdamW/OneCycle in legacy/native runs); validation mAUC selection |
| `CIPL` | Imports official `CIPL` model | Class-specific prototypes plus normal class, paired cross-attention, prediction/interpretation distillation, EMA teacher, cluster/separation loss, training-only prototype projection and last-layer refinement | Shared ResNet-50; supported bottleneck addon instead of the upstream regular addon that hardcodes 256 input channels; fix the joint-loss off-by-one that omits a disease; no test-set threshold tuning |
| `ProCI` | Imports official generic backbone wrapper, memory, attention, causal module and SAM optimizer | Class-specific features, confidence-gated prototype memory, SCU and causal auxiliary losses, scheduled weighting, SAM, EMA confidence, classification-only inference | Shared ResNet-50 and reference classification loss; single GPU without distributed synchronization; cloned pre-write snapshots fix autograd mutation hazards; choose student/EMA on validation |
| `MICCAI2024` | Imports official `Qeruy2Label_New` and `DagmaMLP` directly | Disease queries, transformer, feature reconstruction, log-det DAG penalty, classification loss | Shared backbone and common data loader; avoids loading the entire old MMPretrain fork; standard SGD warmup/cosine schedule; restores global default dtype after the upstream constructor |
| `BankCausal` | Official MMEngine classifier/head/loss/SwAV hook | Two training views, prototype queue and prior, warmup, memory intervention, loss decay, EMA hook | Shared ResNet-50/loss/optimizer/scheduler; current annotations/preprocessor; deterministic eval crop; common metrics; saved non-buffer inference state; no forced intervention-stage change at test time |

ProCI retains the upstream single-device two-stage prototype update; SAM's second
pass does **not** update memory again. Its causal module currently normalizes a
prior argument but does not use that argument in the attention calculation; the
adapter does not invent a different implementation. CIPL's added normal class is
internal only; reported predictions retain exactly the dataset's disease order.

Sources: [CIPL](https://github.com/cwangrun/CIPL),
[MLC-PAT](https://github.com/xiemk/MLC-PAT),
[ProCI](https://github.com/JustinLiam/ProCI),
[MICCAI2024](https://github.com/davelailai/multi-disease-detection-guided-by-causal-estimation),
[BankCausal](https://github.com/davelailai/BankCausal).
The inspected checkouts do not contain a top-level license file. They are fetched
separately rather than redistributed as this project's code; check upstream
licensing before publishing a package that bundles their source.

## Setup

Run from this project's root with the existing `mmcls` environment:

```bash
python -m pip install -r tools/SOTA/requirements.txt
python tools/SOTA/fetch_sources.py
python -B tools/SOTA/test_adapters.py
python -B tools/SOTA/test_protocol.py
```

Existing checkouts must match their pinned commit and have no modified tracked
files. The fetch script never resets or overwrites an existing checkout.

## Training

```bash
python -B tools/SOTA/train.py \
  --method MLC-PAT --dataset FFA \
  --image-root /nobackup/projects/bdliv04/jxie/SCIF/data \
  --work-dir work_dirs/SOTA_aligned/FFA/MLC-PAT/seed0 --seed 0
```

Use `CIPL`, `ProCI`, or `MICCAI2024` with the same command. For BankCausal:

```bash
python -B tools/SOTA/prepare_prior.py \
  --dataset FFA --image-root /nobackup/projects/bdliv04/jxie/SCIF/data \
  --work-dir work_dirs/SOTA/FFA/BankCausal/seed0 --seed 0
python -B tools/SOTA/train_bankcausal.py \
  work_dirs/SOTA/FFA/BankCausal/seed0/resolved.py
```

Supported datasets: `FFA`, `Endo`, `ODIR`. Endo defaults to
`patient_disjoint_balanced_seed42`, never the old image-level split. ODIR defaults
to `onsite_test.pkl`; `--test-ann` can explicitly choose another test set.
`--train-ann`, `--val-ann`, `--test-ann`, and `--annotation-root` override paths.
The image root is the directory **containing** `FFA`, `Endo`, and `OIA-ODIR`.
It overrides old absolute image paths containing a `data` path component.

`configs/SOTA/methods.json` stores the prior/native adaptation defaults; the aligned
mode overrides eligible epoch budgets from the reference config. CIPL's staged
budget remains an explicit exception. Do not claim identical optimizer settings
or identical compute. Tune learning rates and auxiliary weights only on validation
with a predeclared comparable search budget; matching a loss alone is not tuning.

Each run chooses its best checkpoint using **validation mAUC only**. Thresholded
metrics use 0.5, with no test-label threshold fitting. Test evaluation occurs only
after model selection. Outputs include `pred_results.pkl` and `test_metrics.json`;
the standalone runner also writes `run.json`, `history.jsonl`, and `best.pth`.
All metric values are fractions, not percentages. Prediction records contain
image path, target scores, prediction scores, and patient group ID when available.
Group checks can only detect patient overlap when annotations contain group IDs;
they do not replace validation of the split manifests.

### Local Evaluation of Downloaded Runs

```bash
python -B tools/SOTA/evaluate_suite.py --dry-run
python -B tools/SOTA/evaluate_suite.py
```

The default run tree is the sibling `Prototype-TDE/work_dirs/SOTA_aligned`.
Override `--root`, `--reference-root`, and `--image-root` for another layout.
Use `--dataset Endo` or `--method ProCI` to restrict the selection. Evaluation
uses only the saved validation-best checkpoint and writes into each run's
`eval_local`, leaving the training-time test outputs intact. Missing or ambiguous
checkpoints are reported, not substituted with last-epoch checkpoints. Migrated
annotation files must retain ordered image identities, labels, and patient groups.
The root-level `local_test_metrics.csv` and `local_test_results.md` aggregate
successful local evaluations; the latter reports sample SD across random seeds.
`--summarize-only` rebuilds these tables without inference. BankCausal also accepts
`train_bankcausal.py CONFIG --eval CHECKPOINT` for standalone test-only execution.

For Bede, activate `mmcls` before submission:

```bash
sbatch scripts/sota_bede.sh
```

The array has 15 tasks: five methods times three datasets. Each task runs seeds
0-4 sequentially, for 75 runs total, without an array concurrency limit. No jobs
are submitted by installing these files. Override `BATCH_SIZE` or `IMAGE_ROOT`
through exported environment variables. New jobs default to aligned mode and
`work_dirs/SOTA_aligned`; `SOTA_PROTOCOL=native` selects a separate output tree.
The default physical batch is now 32. Check memory usage before full runs,
especially CIPL and PAT's five-view batches. If it does not fit, predeclare a
smaller common batch for the relevant comparisons, including Q2L/Q2L-TDE, rather
than silently lowering only one method's batch.

## Verification

```bash
python -B tools/SOTA/test_training_smoke.py
python -B tools/SOTA/test_bankcausal_smoke.py
```

These tests create temporary synthetic images, train for one tiny epoch without
pretraining downloads, reload the validation-selected checkpoint and export
predictions. Unit tests additionally cover CIPL joint distillation/projection and
ProCI post-warmup memory/backward behavior. They are not evidence of benchmark
accuracy or clinical validity. GPU execution and full training remain to be
validated on Bede. The training entrypoint does not implement interrupted-run
resume; use a fresh output directory for each complete run.
