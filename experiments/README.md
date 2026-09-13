# What is in this folder

Every number in the report comes from a file here. Nothing here needs a GPU to
read, so every figure can be redrawn from these CSVs alone (notebook cell 30).

## The results — these are the evidence

| File | What it holds |
|---|---|
| `history_<run>.csv` | one row per epoch: training loss, validation Dice per structure, mean. Five files, one per run. |
| `test_<run>.csv` | one row per test patient per structure: Dice, HD95, HD max, surface Dice @1 mm and @2 mm, and the empty-mask flags. Five files. |
| `test_all_runs.csv` | the five `test_` files concatenated. Derived — regenerate it, do not edit it. |
| `summary_by_run.csv` | mean of `test_all_runs.csv` grouped by run and structure. **This is the results table in the report.** Derived. |
| `test_by_batch.csv` | the same test results split by acquisition batch (cases 01–20 vs 21–42), to check whether the label batch effect shows up in the results. Derived. |

`history_ce/dice/compound/tversky.csv` were reconstructed from a printed epoch
log at three-decimal precision — see `RECOVERED.md` before quoting them.

## Context — cited in method and limitations

| File | What it holds |
|---|---|
| `preprocessing_1.csv` | per-case preprocessing output: slice counts, voxel counts per structure, class balance. The batch-effect statistics come from here. |
| `crop_losses.csv` | how many label voxels the 256×256 crop clipped, per case. Limitations. |
| `baseline_9cases.csv` | the two baselines on the current 9 test patients: all-background and HU threshold. |

## Superseded — kept for the record, not cited

`baseline_4cases.csv` and `metrics_1.csv` are from the earlier 20-case split.
`train_run_1.csv` is the very first training run (28 epochs, compound loss),
before the five-run comparison. `results.csv` is an early stub.

These are kept because the implementation log refers to them by date and
because deleting evidence of how the project developed would be worse than a
slightly untidy folder. They are not part of the results.
