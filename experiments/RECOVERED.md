# Recovered histories — read this before quoting these files

`history_ce.csv`, `history_dice.csv`, `history_compound.csv` and
`history_tversky.csv` were **reconstructed from the printed epoch log** of the
Colab session of 10 Sep 2026, not written by `fit()`. The original CSVs were
written into the Colab checkout and lost when the runtime was recycled.

What that means for the numbers:

- Dice values are **rounded to three decimals**, because that is all the log printed. The curves are exact to that precision; the "best epoch" marked on a figure can be a epoch or two off where two rounded values tie.
- Every row was checked: the printed `mean` equals the mean of the four printed structure columns in all 116 rows, so nothing was mis-parsed.
- `history_tversky.csv` has 26 rows (epochs 0-25) because the run was killed at epoch 25. That run has no saved weights and no test CSV.
- `ce`, `dice` and `compound` reached epoch 29 and their best weights survived in Drive, so their test CSVs can be regenerated exactly (notebook cell 27).

Say this in the journal. Reconstructed evidence is fine; reconstructed evidence presented as original is not.
