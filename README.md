# Segmenting Cochleae in head-and-neck CT

Cochleae and parotid glands in planning CT, using a 2-D U-Net written from
scratch in PyTorch. Assignment 2, Machine Learning and Advanced Data Analytics,
UTS.

The point of the project is the contrast between the two structures. A parotid
gland is roughly 26,000 voxels; a cochlea is about 67 — four voxels across at
1 mm. Same scan, same model, same training, two completely different outcomes,
and the interesting question is which design choices decide that.

**Colab notebook (self-contained):** `https://colab.research.google.com/github/gs0232/ml-a2-hanseg/blob/main/notebooks/a2_main.ipynb`

---

## Result

Five training runs, identical in everything except the loss. Mean Dice over the
9 held-out patients, left and right pooled:

| run | loss | cochlea | parotid | complete misses (of 36) |
|---|---|---|---|---|
| `ce` | cross-entropy | **0.626** | 0.793 | **0** |
| `dice` | soft Dice | 0.000 | 0.810 | 9 |
| `compound` | cross-entropy + soft Dice | 0.000 | 0.813 | 18 |
| `tversky` | Tversky, α 0.7 / β 0.3 | 0.000 | 0.796 | 9 |
| `compound_over6` | compound, cochlea slices ×6 | 0.000 | 0.787 | 18 |

Two things fall out of that table.

**The choice of loss is irrelevant for the easy structure and decisive for the
hard one.** The parotid spans 0.787–0.813 across all five runs; only one of ten
pairwise comparisons survives correction for multiple testing, and the largest
gap is 0.026 Dice. The cochlea spans 0.626 to 0.000.

**The cochlea failure is not class imbalance.** `compound_over6` raised the
fraction of training slices containing a cochlea from 8.6% to 36% and changed
nothing — same 0.000, same 18 complete misses. What separates the runs is the
shape of the loss, not how often the structure is shown.

### Dice is the wrong metric for this application

The `ce` run, same predictions scored four ways:

| | Cochlea L / R | Parotid L / R |
|---|---|---|
| Dice ↑ | 0.597 / 0.655 | 0.777 / 0.809 |
| Surface Dice @ 1 mm ↑ | **0.741 / 0.822** | 0.487 / 0.551 |
| HD95 ↓ | **2.16 / 1.93 mm** | 6.94 / 5.36 mm |
| HD max ↓ | **2.79 / 2.79 mm** | 68.8 / 99.3 mm |

The ranking reverses. Dice is scale-sensitive — one voxel of boundary error
costs a large fraction of a 67-voxel cochlea and almost nothing on a parotid —
while the parotid predictions carry isolated stray voxels up to 99 mm away that
Dice barely registers. For surgical planning near the cochlea, the worst-case
boundary distance is the number that matters, and Dice hides it.

---

## Reproducing this

### Fast path (3 min) — every figure, no GPU

All results are committed as CSVs, so the figures can be rebuilt without a GPU,
without the 4.9 GB dataset and without the trained weights.

Open the Colab notebook and run cells **1, 2, 3, 16, 22, 30**.

### Full path (4.5 h)

Run the notebook top to bottom: download (4.9 GB), preprocess 42 patients
(~40 min), five training runs (~3.5 h on a T4). Nothing needs to be set up by
hand; the notebook clones this repository and installs what it needs.

### Tests

```bash
pytest -q
```

58 tests. 14 of them need PyTorch, so on a machine without it pytest reports
`44 passed, 1 skipped` — that skip is the whole torch module, not a failure.
Notebook cell 32 runs the full suite in Colab, where torch is present.

---

## Data

**HaN-Seg** — 42 patients, CT and T1 MR, 30 expert-contoured organs-at-risk.
<https://zenodo.org/records/7442914> · 4.9 GB · CC BY-NC-ND 4.0 ·
MD5 `c3d3070de3034933a63031b73b9cf0fc`

This project uses the CT only, and four of the thirty structures: `Cochlea_L`,
`Cochlea_R`, `Parotid_L`, `Parotid_R`.

---

## How it works

**Preprocessing** — resample to 1 mm isotropic (nearest-neighbour for masks,
linear for the CT), crop 256 × 256 around the head centre, and build two input
channels: a soft-tissue window (level 40 / width 400) and a bone window
(400 / 1800). Neither window alone shows both structures. Cached as one `.npz`
per patient.

**Slice selection** — every slice containing a target structure plus an equal
number of random empty slices. This creates a deliberate mismatch between
training and deployment, so all evaluation is done on complete volumes.

**Split** — patient-level, 25 / 8 / 9, seed 0, frozen in `src/split.json` before
the first training run and never regenerated. Splitting by slice instead of by
patient would put near-copies of training data in the test set.

**Model** — `UNet2D`, 2 input channels, 5 output classes (background + four
structures), channels 32 → 512, **7,762,885 parameters**. The parameter count is
independent of image size.

**Training** — 30 epochs, AdamW at 1e-3 with cosine annealing, batch size 8,
mixed precision. The weights kept are those from the epoch with the highest mean
validation Dice, chosen on validation, never on test.

**Evaluation** — Dice, HD95, HD max and surface Dice at 1 mm and 2 mm, computed
per structure per patient on the full 3-D volume, then averaged. Never pooled
across patients, never averaged across structures. Two baselines: predicting
background everywhere (99.8% pixel accuracy, 0.00 Dice) and a Hounsfield-unit
threshold.

---

## Layout

```
src/          data.py preprocessing · dataset.py the split · loader.py batching
              model.py the U-Net · losses.py the four losses · metrics.py Dice,
              Hausdorff, surface Dice · train.py the training loop
              experiment.py one run end-to-end · viz.py every figure
              baselines.py · split.json (the frozen split)
tests/        58 tests
notebooks/    a2_main.ipynb — the submitted notebook
experiments/  all results as CSV. See experiments/README.md
figures/      every figure in the report, rebuildable by cell 30
journal/      implementation log, AI usage, knowledge gaps
```

Trained weights (`.pt`, ~30 MB each) are **not** in the repository. They live in
Google Drive. Every cell that needs them checks first and skips with a message,
so the notebook still works for someone who has only this repository.

---

## Limitations

**The ground truth is not internally consistent.** Cochlea contour volume
roughly doubles between patients 01–20 (median 101 voxels) and 21–42 (median
196). Parotid volume changes by 1.09× and not significantly (p = 0.062). A changepoint search over 31 candidate positions picks case 21. This puts a ceiling on achievable cochlea Dice that has nothing to do with the model. `experiments/test_by_batch.csv` splits the results by batch. An untested alternative explanation is that later scans were acquired at finer resolution.

**One seed per configuration.** Differences smaller than seed-to-seed variation
cannot be distinguished, and that variation was not measured. Small differences
between runs are not claimed as real.

**Four of the five history CSVs were reconstructed** from a printed epoch log at
three-decimal precision after a Colab disconnect destroyed the originals. See
`experiments/RECOVERED.md` before quoting them. The test CSVs were regenerated
exactly from the surviving weights.

**The contours were drawn for radiotherapy planning, not surgery.** HaN-Seg was
annotated by radiation oncologists for dose planning. Using it to argue about
operative planning is a proxy and is stated as one.

**2-D, not 2.5-D or 3-D.** The model sees one axial slice at a time and cannot
use the slices above and below, which is most costly for the cochlea, it spans
only a handful of slices.
