# Additional Comparison Methods

These are **controlled medical-dataset adaptations**, not claims to reproduce
the original papers' benchmark numbers. All methods use the existing dataset
annotations, ResNet-50, the project's SwAV initialization, and 448-pixel inputs.
The source revisions are pinned in `sources.json`. No training results have been
generated on the full medical datasets yet.

## Aligned Comparison Protocol (Default)

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
- Method-specific optimizers, schedules, EMA rules and auxiliary objectives.
  These remain validation-tunable and are NOT claimed to be identical.

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
| `MLC-PAT` | PAT-T cross-attention, implemented from the upstream tensor operations | Full image plus 2x2 patches, three classifiers, patch-logit softmax attention, full/patch ASL, averaged-logit inference, EMA | ResNet-50 instead of the original larger backbones; reference classification loss and image operations in aligned mode; AdamW; validation mAUC instead of mAP selection |
| `CIPL` | Imports official `CIPL` model | Class-specific prototypes plus normal class, paired cross-attention, prediction/interpretation distillation, EMA teacher, cluster/separation loss, training-only prototype projection and last-layer refinement | Shared ResNet-50; supported bottleneck addon instead of the upstream regular addon that hardcodes 256 input channels; fix the joint-loss off-by-one that omits a disease; no test-set threshold tuning |
| `ProCI` | Imports official generic backbone wrapper, memory, attention, causal module and SAM optimizer | Class-specific features, confidence-gated prototype memory, SCU and causal auxiliary losses, scheduled weighting, SAM, EMA confidence, classification-only inference | Shared ResNet-50 and reference classification loss; single GPU without distributed synchronization; cloned pre-write snapshots fix autograd mutation hazards; choose student/EMA on validation |
| `MICCAI2024` | Imports official `Qeruy2Label_New` and `DagmaMLP` directly | Disease queries, transformer, feature reconstruction, log-det DAG penalty, classification loss | Shared backbone and common data loader; avoids loading the entire old MMPretrain fork; standard SGD warmup/cosine schedule; restores global default dtype after the upstream constructor |
| `BankCausal` | Official MMEngine classifier/head/loss/SwAV hook | Two training views, prototype queue and prior, warmup, memory intervention, loss decay, official optimizer/schedule and EMA hook | Shared ResNet-50; current annotations/preprocessor; deterministic eval crop; common metrics; save non-buffer inference state and do not force a different intervention stage at test time |

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
