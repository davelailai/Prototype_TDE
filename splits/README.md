# Fixed dataset splits

Each dataset has a `split_manifest.csv` containing split membership, anonymous
patient/group IDs, image paths relative to the external data root, and binary
labels in the original class order. `split_report.json` records counts, source
checksums, manifest checksum, grouping provenance, and overlap checks.

These files do not include images or machine-specific absolute paths.

| Dataset | Assignment | Train patients | Validation patients | Test patients |
| --- | --- | ---: | ---: | ---: |
| FFA | Existing assignments | 350 | 50 | 100 |
| Endo | Revised balanced patient split, seed 42 | 56 | 8 | 16 |
| ODIR | Existing assignments | 2800 | 700 | 1000 on-site; 500 off-site |

All pairwise patient and image intersections are zero under the documented ID
definitions. Endo study IDs identify distinct patients according to the dataset
owner's confirmation. FFA IDs come from `original_case`; ODIR IDs come from the
numeric prefix shared by left/right eye filenames. This audit does not establish
identical patient membership to a published paper's split.

## Restore on Bede

From the project root, with a fresh local `data/` annotation destination:

```bash
python tools/Analysis/dataset_splits.py materialize \
  --image-root /nobackup/projects/bdliv04/jxie/SCIF/data \
  --out-root data \
  --relative-images
```

The image root must contain `FFA/fa_data`, `Endo/Endo/images`, and the
`OIA-ODIR` image folders. No image files are copied. All inputs are validated
before any PKLs are written; existing annotation files are never overwritten.
For an existing checkout with annotations, choose a fresh `--out-root`, such as
`data_restored`, and override the corresponding `ann_file` config values.

Output paths match the checked-in dataset configs:

```text
data/FFA/{train,val,test}_class6.pkl
data/Endo/Endo/patient_disjoint_balanced_seed42/{train,val,test}_list.pkl
data/OIA-ODIR/{train,val,onsite_test,offsite_test}.pkl
```

With `--relative-images`, each generated PKL stores paths relative to the image
root. Supply that root when submitting a training job (the same overrides work
for FFA, Endo and ODIR):

```bash
python tools/Train.py configs/Endo/Endo_CounterFactual_TDE.py \
  --work-dir work_dirs/Endo_patient_disjoint_run0 \
  --cfg-options \
  train_dataloader.dataset.data_prefix.img_path=/nobackup/projects/bdliv04/jxie/SCIF/data \
  val_dataloader.dataset.data_prefix.img_path=/nobackup/projects/bdliv04/jxie/SCIF/data \
  test_dataloader.dataset.data_prefix.img_path=/nobackup/projects/bdliv04/jxie/SCIF/data
```

For testing, pass `test_dataloader.dataset.data_prefix.img_path` to
`tools/Test.py --cfg-options`. Never use relative-image PKLs without a prefix.
Alternatively, omit `--relative-images` to write absolute paths based on the
provided `--image-root`; then no additional image prefix is needed.

All methods must use the same manifests. Endo models trained on the old
image-level split must be retrained for the revised patient-disjoint evaluation.

## Re-export local annotations

```bash
python tools/Analysis/dataset_splits.py export --manifest-root splits_new
```

Export reads the annotations already assigned to each split; it does not
resample FFA or ODIR. Materialization reads the committed manifests directly
and never reruns random splitting. PKLs and original datasets remain excluded
from Git; only the portable manifests and reports are committed.
