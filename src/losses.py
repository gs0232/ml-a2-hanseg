"""Training losses.

There is only ONE implementation here. Dice is Tversky with alpha = beta =
0.5, so it is written as a wrapper rather than copied — two copies of the same
formula drift apart the first time one of them is edited.
"""
import torch
import torch.nn.functional as F


def tversky_loss(logits, target, alpha: float = 0.5, beta: float = 0.5,
                 eps: float = 1.0, ignore_background: bool = True):
    """Soft Tversky loss. alpha weights false negatives, beta false positives.

    alpha = beta = 0.5 gives Dice, which treats the two errors as equally bad.
    They are not equally bad here: under-segmenting a nerve tells a surgeon
    there is clearance where there is none, while over-segmenting only makes
    them drill conservatively. alpha > beta encodes that.

    About eps, which is NOT merely a divide-by-zero guard. Softmax never
    returns exactly zero, so a structure that is absent from both the label
    and the prediction still accumulates roughly 1e-7 of false-positive mass
    across 65k pixels. Measured, for a PERFECT prediction with three absent
    classes:

        eps      index of an absent class     total loss
        0                        0.000            0.750
        1e-8                     0.000            0.750
        1e-2                     0.974            0.020
        1.0                      1.000            0.0002

    With eps below that noise floor the ratio is decided by rounding error and
    a perfectly correct prediction scores as badly wrong. eps must be of order
    one so it swamps the noise. That is why it is 1.0.
    """
    probs = logits.softmax(dim=1)
    onehot = F.one_hot(target.long(), logits.shape[1]).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    tp = (probs * onehot).sum(dims)
    fn = ((1.0 - probs) * onehot).sum(dims)
    fp = (probs * (1.0 - onehot)).sum(dims)
    index = (tp + eps) / (tp + alpha * fn + beta * fp + eps)
    if ignore_background:
        index = index[1:]          # background is 99.8% of voxels; including
                                   # it would let a blank prediction score well
    return 1.0 - index.mean()


def dice_loss(logits, target, eps: float = 1.0, ignore_background: bool = True):
    """Soft Dice. Exactly Tversky at alpha = beta = 0.5."""
    return tversky_loss(logits, target, 0.5, 0.5, eps, ignore_background)


def compound_loss(logits, target, w_ce: float = 1.0, w_dice: float = 1.0):
    """Cross-entropy plus Dice, the standard pairing for medical segmentation.

    Cross-entropy gives well-behaved per-pixel gradients early on, when the
    predictions are so bad that Dice is almost flat. Dice supplies the class
    balance that cross-entropy lacks: on its own, cross-entropy is minimised
    fairly well by predicting background everywhere, because background really
    is 99.8% of the answer.
    """
    return w_ce * F.cross_entropy(logits, target.long()) + w_dice * dice_loss(logits, target)


def get_loss(name: str):
    """Name -> callable(logits, target) -> scalar.

    Selecting by string keeps the experiment honest: the same training code
    runs for every arm of the loss comparison, and the name is what gets
    written into results.csv.
    """
    table = {
        "ce": lambda l, t: F.cross_entropy(l, t.long()),
        "dice": dice_loss,
        "compound": compound_loss,
        "tversky": lambda l, t: tversky_loss(l, t, alpha=0.7, beta=0.3),
    }
    if name not in table:
        raise ValueError(f"unknown loss {name!r}; choose from {sorted(table)}")
    return table[name]
