"""Figures for the report. One figure per PNG, on purpose.

Every function reads a CSV that a run wrote and returns a matplotlib figure,
so nothing here needs a GPU or a trained model — you can redraw any figure
months later from the CSVs alone, which is why the CSVs are committed to
experiments/ and the .pt weights are not.

Two colour conventions, kept apart:

  * Figures that compare STRUCTURES use two hues — blue for cochlea, orange
    for parotid — and carry left/right in the line style. Two hues instead of
    four, because left and right of the same organ are the same kind of thing.
  * Figures that compare RUNS use four hues, one per run.

Both palettes were checked for colour-blind separation before use. Do not
swap in prettier colours without re-checking: "these look different to me" is
not the same as "these look different to a protanope".
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

INK, INK_2, GRID = "#0b0b0b", "#3d3d3b", "#d7d7d3"

# structure palette (2 hues; left/right by line style)
COCHLEA, PAROTID = "#2a78d6", "#eb6834"
STRUCTURES = [("Cochlea_L", COCHLEA, "-"), ("Cochlea_R", COCHLEA, "--"),
              ("Parotid_L", PAROTID, "-"), ("Parotid_R", PAROTID, "--")]

# run palette (fixed order — never cycled, never re-assigned when a run drops out)
RUN_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

DICE_LABEL = "Dice coefficient (0–1, dimensionless)"


def _style():
    """Large type, a light grid on both axes, a framed legend, white paper.

    Sized for a figure that will be shrunk to about half a page in the report:
    if the tick labels are comfortable on screen they will be unreadable in
    print, so they are deliberately too big here.
    """
    mpl.rcParams.update({
        "figure.figsize": (11, 5.6), "figure.dpi": 110,
        "font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.9,
        "axes.edgecolor": "#9a9a96", "axes.linewidth": 0.9,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "legend.frameon": True, "legend.framealpha": 1.0,
        "legend.edgecolor": "#b4b4b0", "legend.borderpad": 0.6,
    })


def _titles(ax, main, sub):
    """A short title that says what the figure is, and a smaller line under it
    that says which run it came from. Two lines rather than one long one: a
    figure pasted into a report without its caption still has to identify
    itself, but a 110-character title wraps badly at report width."""
    ax.set_title(main, loc="center", pad=30)
    if sub:
        ax.text(0.5, 1.015, sub, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=12, color=INK_2)


def _save(fig, out_path):
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
        print("wrote", out_path)
    return fig


# ===========================================================================
def plot_loss(csv_path, out_path=None, run_label="", loss_name="compound"):
    """Training loss against epoch, for ONE run.

    Its own PNG rather than a panel beside the Dice curve. A loss and a Dice
    score have different units and opposite directions of "good"; the only way
    to put them in one frame is two y-scales, and a two-scale chart lets you
    place the crossing point wherever you like by choosing the scales. Two
    files instead.
    """
    df = pd.read_csv(csv_path)
    best = int(df.val_mean_dice.idxmax())
    _style()
    fig, ax = plt.subplots()

    ax.plot(df.epoch, df.train_loss, color=INK, lw=2.2, solid_capstyle="round",
            label=f"Training loss ({loss_name})")
    ax.axvline(best, color="#c0392b", lw=1.6, ls="--",
               label=f"Best epoch on validation ({best})")
    ax.annotate(f"{df.train_loss.iloc[-1]:.3f}",
                xy=(df.epoch.iloc[-1], df.train_loss.iloc[-1]),
                xytext=(-6, 12), textcoords="offset points", ha="right",
                fontsize=12, color=INK_2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (dimensionless)")
    ax.set_ylim(0, max(df.train_loss) * 1.12)
    _titles(ax, "Training loss", run_label)
    ax.legend(loc="upper right")
    return _save(fig, out_path)


# ===========================================================================
def plot_val_dice(csv_path, out_path=None, run_label=""):
    """Validation Dice against epoch, one line per structure, for ONE run."""
    df = pd.read_csv(csv_path)
    best = int(df.val_mean_dice.idxmax())
    _style()
    fig, ax = plt.subplots()

    for name, colour, style in STRUCTURES:
        ax.plot(df.epoch, df[f"val_dice_{name}"], color=colour, lw=2.2,
                ls=style, solid_capstyle="round", label=name.replace("_", " "))
    ax.axvline(best, color="#c0392b", lw=1.6, ls="--",
               label=f"Best epoch ({best})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(DICE_LABEL)
    ax.set_ylim(-0.03, 1.05)
    _titles(ax, "Validation Dice per structure", run_label)

    ax.legend(loc="center right", ncol=1, bbox_to_anchor=(1.0, 0.42))
    return _save(fig, out_path)


# ===========================================================================
def plot_val_curves(histories, out_path=None,
                    title="Validation mean Dice — all runs"):
    """One line per run: does a loss get further, or only get there sooner?

    `histories` is {run_name: csv_path}, in the order you want them coloured.
    """
    _style()
    fig, ax = plt.subplots(figsize=(12.6, 5.6))
    labels = []
    for i, (name, path) in enumerate(histories.items()):
        df = pd.read_csv(path)
        ax.plot(df.epoch, df.val_mean_dice, color=RUN_COLOURS[i % len(RUN_COLOURS)],
                lw=2.2, solid_capstyle="round",
                label=f"{name}  (best {df.val_mean_dice.max():.3f})")
        labels.append([float(df.val_mean_dice.iloc[-1]), name])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Dice over the four structures (0–1)")
    ax.set_ylim(0, max(y for y, _ in labels) * 1.4)
    _titles(ax, title, "one line per run; the label gives each run's best epoch")

    # Direct labels as well as the legend: two of these hues sit under 3:1
    # against white paper, so identity must not rest on colour alone. Runs that
    # finish within a hair of each other would print on top of one another, so
    # push them apart from the top down.
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.06
    labels.sort(key=lambda t: -t[0])
    for j in range(1, len(labels)):
        labels[j][0] = min(labels[j][0], labels[j - 1][0] - gap)
    x_end = pd.read_csv(next(iter(histories.values()))).epoch.iloc[-1]
    for y, text in labels:
        ax.annotate(text, xy=(x_end, y), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=12, color=INK)

    ax.legend(loc="lower right")
    fig.tight_layout(rect=[0, 0, 0.84, 1])
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
        print("wrote", out_path)
    return fig


# ===========================================================================
def plot_run_comparison(test_csv, out_path=None, order=None,
                        title="Test Dice by run — 9 held-out patients"):
    """Grouped bars: what each run actually scored on the test patients.

    Back to the structure palette here. The question this figure answers is
    "did the cochlea ever move?", so cochlea and parotid must stay the two
    things you can tell apart at a glance.
    """
    df = pd.read_csv(test_csv)
    df["organ"] = df.structure.str.split("_").str[0]
    means = df.groupby(["run", "organ"]).dice.mean().unstack()
    if order:
        means = means.loc[[o for o in order if o in means.index]]
    _style()
    fig, ax = plt.subplots()
    x = range(len(means))
    w = 0.32
    for off, organ, colour in ((-w / 2 - 0.02, "Cochlea", COCHLEA),
                               (w / 2 + 0.02, "Parotid", PAROTID)):
        vals = means[organ].values
        ax.bar([i + off for i in x], vals, width=w, color=colour,
               label=organ)
        for i, v in zip(x, vals):
            ax.annotate(f"{v:.3f}", xy=(i + off, v), xytext=(0, 6),
                        textcoords="offset points", ha="center",
                        fontsize=12, color=INK_2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(means.index, color=INK)
    ax.set_xlabel("Run")
    ax.set_ylabel("Mean Dice on the test patients (0–1)")
    # headroom for the legend, with the ticks stopping at 1.0 so the axis still
    # reads as the 0-1 range a Dice score actually lives in
    ax.set_ylim(0, 1.34)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="x", visible=False)
    _titles(ax, title, "3-D Dice on the full volumes, left and right pooled")
    ax.legend(loc="upper right")
    return _save(fig, out_path)
