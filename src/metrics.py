"""Segmentation metrics, computed per case in 3D.

Two families, and the difference between them is what this project argues about:

  dice()          counts how many voxels overlap
  hd()            measures how far apart the two boundaries are, in millimetres
  surface_dice()  what fraction of the boundary is within a tolerance

All distances are in millimetres. That only works because the volumes were
resampled to 1 mm isotropic in preprocessing — on the scanner's own
anisotropic grid, a millimetre would mean something different in each
direction and these numbers would be meaningless.

NOTE on `spacing`: it is in ARRAY order (z, y, x), not the (x, y, z) that
SimpleITK's GetSpacing() reports. After 1 mm resampling both are (1, 1, 1),
so it does not bite here — but it would on native data.
"""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """2|P and G| / (|P| + |G|). 1.0 identical, 0.0 no overlap.

    Returns NaN when both masks are empty: there is nothing to score, and
    either 0.0 or 1.0 would be a lie about a structure that is not there.
    """
    p, g = np.asarray(pred, bool), np.asarray(gt, bool)
    denom = int(p.sum()) + int(g.sum())
    if denom == 0:
        return float("nan")
    return float(2.0 * np.logical_and(p, g).sum() / denom)


def surface(mask: np.ndarray) -> np.ndarray:
    """Boundary voxels: inside the mask, but with at least one face outside it."""
    m = np.asarray(mask, bool)
    if not m.any():
        return m
    return m & ~binary_erosion(m)


def surface_distances(pred, gt, spacing=(1.0, 1.0, 1.0)):
    """(pred_to_gt, gt_to_pred) distances in mm, one value per surface voxel.

    (None, None) if either mask is empty, because there is no surface to
    measure from.
    """
    sp, sg = surface(pred), surface(gt)
    if not sp.any() or not sg.any():
        return None, None
    to_gt = distance_transform_edt(~sg, sampling=spacing)
    to_pred = distance_transform_edt(~sp, sampling=spacing)
    return to_gt[sp], to_pred[sg]


def hd(pred, gt, spacing=(1.0, 1.0, 1.0), percentile: float = 95.0) -> float:
    """Hausdorff distance in mm at a given percentile of the two-way surface
    distance.

    percentile=100 is the classical Hausdorff distance: the single worst
    boundary error anywhere. It is the number a surgeon cares about, because
    you do not get to discard your worst mistake — but it is also decided by
    one voxel, so it is noisy.

    percentile=95 discards the worst 5%, which makes it stable but ALSO blind
    to any error affecting less than 5% of the boundary. Report both.

    NaN if either mask is empty. A complete miss has no boundary distance, and
    substituting a large arbitrary number would dominate any average, so
    misses are counted separately instead (see evaluate_case).
    """
    a, b = surface_distances(pred, gt, spacing)
    if a is None:
        return float("nan")
    return float(np.percentile(np.concatenate([a, b]), percentile))


def hd95(pred, gt, spacing=(1.0, 1.0, 1.0)) -> float:
    """95th percentile Hausdorff distance in mm."""
    return hd(pred, gt, spacing, 95.0)


def hd_max(pred, gt, spacing=(1.0, 1.0, 1.0)) -> float:
    """Classical Hausdorff distance in mm: the single worst boundary error."""
    return hd(pred, gt, spacing, 100.0)


def surface_dice(pred, gt, spacing=(1.0, 1.0, 1.0), tol_mm: float = 1.0) -> float:
    """Fraction of both surfaces lying within tol_mm of the other surface.

    Reads as "how much of my contour is close enough to be usable", which is
    nearer to the surgical question than overlap is. 1.0 is perfect.
    """
    a, b = surface_distances(pred, gt, spacing)
    if a is None:
        return float("nan")
    return float((np.sum(a <= tol_mm) + np.sum(b <= tol_mm)) / (len(a) + len(b)))


def evaluate_case(pred_lab, gt_lab, class_names, spacing=(1.0, 1.0, 1.0),
                  tols=(1.0, 2.0)) -> list:
    """Every metric for one case, one row per structure.

    Never averages across structures: a cochlea and a parotid differ by more
    than two orders of magnitude in size, so a single mean would be dominated
    by the easy ones and would hide the whole finding.
    """
    rows = []
    for i, name in enumerate(class_names, start=1):
        p, g = np.asarray(pred_lab) == i, np.asarray(gt_lab) == i
        row = {
            "structure": name,
            "gt_voxels": int(g.sum()),
            "pred_voxels": int(p.sum()),
            "dice": dice(p, g),
            "hd95_mm": hd95(p, g, spacing),
            "hdmax_mm": hd_max(p, g, spacing),
            "missed": bool(g.any() and not p.any()),
            "false_positive_only": bool(p.any() and not g.any()),
        }
        for t in tols:
            row[f"sdice_{t:g}mm"] = surface_dice(p, g, spacing, t)
        rows.append(row)
    return rows
