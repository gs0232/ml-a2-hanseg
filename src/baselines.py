"""Two predictions that need no learning at all.

Their job is to say what a score is worth. A Dice of 0.6 means nothing until
you know what predicting nothing scores.
"""
import numpy as np


def predict_all_background(img: np.ndarray) -> np.ndarray:
    """Predict class 0 everywhere. (n, 2, H, W) -> (n, H, W).

    Scores about 99.8% pixel accuracy on this data and exactly 0.00 Dice on
    every structure. That pair of numbers is the reason accuracy is never
    reported for this task, and the reason the loss function needs designing
    rather than picking.
    """
    return np.zeros((img.shape[0], img.shape[2], img.shape[3]), dtype=np.uint8)


def predict_bone_threshold(img: np.ndarray, thresh: int = 200,
                           cls: int = 1) -> np.ndarray:
    """The obvious hand-written rule: call every bright voxel in the bone
    window a cochlea.

    It cannot work, and why it cannot work is the argument for learning. The
    cochlea is a fluid-filled cavity INSIDE the densest bone in the body, so
    intensity is anti-correlated with the target: the rule finds the otic
    capsule around the cochlea rather than the cochlea itself. No threshold
    fixes that, because the information is not in the intensity of a single
    voxel — it is in the shape of what surrounds it.
    """
    return np.where(img[:, 1] > thresh, cls, 0).astype(np.uint8)
