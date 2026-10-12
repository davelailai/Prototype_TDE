# Prototype-TDE for Multi-Label Medical Image Diagnosis

This repository contains the Prototype-TDE implementation for multi-label medical image diagnosis experiments. The code is organized as a compact reproduction project based on MMPreTrain/MMEngine.

## Overview

Prototype-TDE learns disease-specific representations and prototypes, then estimates the total direct effect (TDE) by prototype-level intervention.

Supported experiment configs:

- `configs/ODIR/ODIR_CounterFactual_TDE.py`
- `configs/FFA/FFA_CounterFactual_TDE.py`
- `configs/Endo/Endo_CounterFactual_TDE.py`

## Main Files

- Prototype-TDE head: `models/heads/Multi_label_Counterfactual_TDE.py`
- Multi-task head wrapper: `models/heads/multi_label_muti_task_head.py`
- Warmup/TDE hooks: `models/hooks/Causal_hook.py`
- Multi-label loss: `models/losses/MILAsyLoss.py`
- Metrics: `evaluation/Rankingmulti_label.py`
- Training entry: `tools/Train.py`
- Test entry: `tools/Test.py`
- Ablation summaries: `docs/ablation/`

## Installation

Create an environment with Python 3.9 or newer, then install PyTorch and OpenMMLab dependencies.

```bash
pip install -U openmim
mim install "mmengine>=0.10.0" "mmcv>=2.0.0" "mmpretrain>=1.0.0rc8"
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version before running experiments.

## Data Preparation

Datasets are not included. Put annotation files under `data/` following the paths used by the configs.

Example layout:

```text
data/
  OIA-ODIR/
    train.pkl
    val.pkl
    onsite_test.pkl
  FFA/
    train_class6.pkl
    val_class6.pkl
    test_class6.pkl
  Endo/
    train.pkl
    val.pkl
    test.pkl
```

If your data is stored elsewhere, update the annotation paths in the corresponding config or override them with `--cfg-options`.

## Training

Single GPU:

```bash
bash scripts/train_prototype_tde.sh configs/ODIR/ODIR_CounterFactual_TDE.py work_dirs/ODIR_Prototype_TDE
```

Multiple GPUs:

```bash
bash tools/dist_train.sh configs/ODIR/ODIR_CounterFactual_TDE.py 4 --work-dir work_dirs/ODIR_Prototype_TDE
```

To run another dataset, replace the config path, for example:

```bash
bash scripts/train_prototype_tde.sh configs/FFA/FFA_CounterFactual_TDE.py work_dirs/FFA_Prototype_TDE
```

## Evaluation

```bash
bash scripts/test_prototype_tde.sh \
  configs/ODIR/ODIR_CounterFactual_TDE.py \
  work_dirs/ODIR_Prototype_TDE/best_multi-class_mAUC_multiclass_epoch_XX.pth \
  work_dirs/ODIR_Prototype_TDE/eval
```

To enable test-time augmentation, append `--tta`.

## Ablation Summaries

The markdown summaries in `docs/ablation/` record margin and lambda ablations for FFA, ODIR, and Endo.

## Notes

- Checkpoints, logs, predictions, TensorBoard events, and `work_dirs*` are ignored by Git.
- Dataset paths are local and should be configured for each machine.
- Pretrained backbones are loaded from public OpenMMLab URLs defined in the configs.
- For reproduction, keep the same image resolution, backbone, warmup settings, intervention margin, and lambda values as the selected config.

