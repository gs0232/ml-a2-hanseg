"""One training run, end to end, written to disk.

Why this file exists: the runs must differ in exactly one thing each. If each
run is typed out by hand in its own notebook cell, they will not — an epoch
count will drift, a seed will be forgotten, and the comparison quietly stops
being a comparison. Everything held fixed is fixed here by construction;
everything that varies is an argument.

Where the outputs go, and why:
    experiments/history_<name>.csv   one row per epoch          -> committed
    experiments/test_<name>.csv      one row per structure/case -> committed
    figures/fig_<name>_*.png         the two per-run figures    -> committed
    <weights_dir>/<name>.pt          the best weights           -> Drive only

The CSVs are small and are the evidence, so they go in git and can be redrawn
into figures at any time without a GPU. The .pt files are ~30 MB each and can
be regenerated, so they stay in Drive and out of the repository.
"""
import os
import numpy as np
import pandas as pd
import torch

from src.loader import make_loaders
from src.model import UNet2D
from src.train import fit, predict_case
from src.metrics import evaluate_case
from src.viz import plot_loss, plot_val_dice

CLASS_NAMES = ("Cochlea_L", "Cochlea_R", "Parotid_L", "Parotid_R")

LOSS_DESC = {
    "ce": "cross-entropy",
    "dice": "soft Dice",
    "compound": "cross-entropy + soft Dice",
    "tversky": "Tversky, alpha 0.7 / beta 0.3",
}


def _label(name, loss_name, epochs, lr, seed, rare_classes, rare_repeat,
           run_index=None, n_runs=None):
    """The subtitle every figure from this run carries, so a PNG that has been
    dragged into a slide deck still says which run produced it."""
    head = f"run {run_index}/{n_runs}: " if run_index else ""
    mix = (f", cochlea slices x{rare_repeat}"
           if rare_classes and rare_repeat > 1 else "")
    return (f"{head}{name} — {LOSS_DESC.get(loss_name, loss_name)}{mix}; "
            f"{epochs} epochs, lr {lr:g}, batch 8, seed {seed}")


def run_one(name, cache_dir, split, device, weights_dir,
            csv_dir="experiments", fig_dir="figures",
            loss_name="compound", rare_classes=(), rare_repeat=1,
            epochs=30, lr=1e-3, batch_size=8, seed=0, augment=True,
            run_index=None, n_runs=None, class_names=CLASS_NAMES):
    """Train one model, evaluate it in 3-D on the test patients, save everything.

    Returns (history_df, test_df).

    The seed is set here rather than left to chance, so two runs that differ
    only in the loss also start from the same initial weights. Without that,
    part of any difference you measure is just a different random start, and
    you cannot tell which part.
    """
    for d in (csv_dir, fig_dir, weights_dir):
        os.makedirs(d, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, val_loader = make_loaders(
        cache_dir, split, batch_size=batch_size, seed=seed, augment=augment,
        rare_classes=rare_classes, rare_repeat=rare_repeat)
    frac = (train_loader.dataset.fraction_containing(rare_classes)
            if rare_classes else None)

    label = _label(name, loss_name, epochs, lr, seed, rare_classes,
                   rare_repeat, run_index, n_runs)
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(f"{len(train_loader.dataset):5d} training slices" +
          (f"   ({frac:.1%} contain a cochlea)" if frac is not None else ""))

    model = UNet2D().to(device)
    history, best = fit(model, train_loader, val_loader, device,
                        loss_name=loss_name, epochs=epochs, lr=lr,
                        class_names=class_names)
    model.load_state_dict(best)
    torch.save(best, f"{weights_dir}/{name}.pt")

    hist_df = pd.DataFrame(history)
    hist_csv = f"{csv_dir}/history_{name}.csv"
    hist_df.to_csv(hist_csv, index=False)

    rows = []
    for cid in split["test"]:
        pred, gt = predict_case(model, cache_dir, cid, device)
        for r in evaluate_case(pred, gt, list(class_names)):
            r.update(case=cid, run=name)
            rows.append(r)
    test_df = pd.DataFrame(rows)
    test_df.to_csv(f"{csv_dir}/test_{name}.csv", index=False)

    plot_loss(hist_csv, f"{fig_dir}/fig_{name}_loss.png",
              run_label=label, loss_name=loss_name)
    plot_val_dice(hist_csv, f"{fig_dir}/fig_{name}_val_dice.png",
                  run_label=label)

    print(test_df.groupby("structure").dice.mean().round(3).to_string())
    return hist_df, test_df


# ---------------------------------------------------------------------------
# The runs.
#
# Runs 1-4 change ONLY the loss. Same cached slices, same split, same seed,
# same optimiser, same schedule, same epoch count — so a difference between
# them was caused by the loss and by nothing else. Tversky is in this group
# because alpha > beta is the loss that matches the surgical argument: missing
# a structure is worse than over-drawing it, and Dice cannot express that.
#
# Run 5 changes the training MIX instead (cochlea slices repeated six times,
# 8.6% -> 36% of the training set). That is a data change, not a loss change,
# so it is not part of the comparison above and must be reported separately.
RUNS = [
    dict(name="ce",             loss_name="ce"),
    dict(name="dice",           loss_name="dice"),
    dict(name="compound",       loss_name="compound"),
    dict(name="tversky",        loss_name="tversky"),
    dict(name="compound_over6", loss_name="compound",
         rare_classes=(1, 2), rare_repeat=6),
]
