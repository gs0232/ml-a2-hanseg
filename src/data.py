import os
import glob
import SimpleITK as sitk
import numpy as np
import warnings

TARGET_FILES = ["Cochlea_L", "Cochlea_R", "Parotid_L", "Parotid_R"]
CROP = 256 # Hyperparamter

def find_one(case_dir: str, pattern: str) -> str:
    """Return the one file in case_dir matching pattern.

    Raises FileNotFoundError naming the case and the pattern unless there is
    exactly one match. Zero means the dataset is not laid out how we assumed;
    two means the pattern is ambiguous. Both should stop the run.
    """
    hits = sorted(glob.glob(os.path.join(case_dir, pattern)))
    if len(hits) != 1:
        raise FileNotFoundError(
            f"{os.path.basename(case_dir)}: expected exactly 1 file matching "
            f"{pattern!r}, found {len(hits)}: "
            f"{[os.path.basename(h) for h in hits]}"
        )
    return hits[0]

def case_files(case_dir: str) -> dict:
    """Locate the files for ONE case. Pure path logic — opens nothing.

    Returns {"CT": path, "Cochlea_L": path, ...}.
    Split out from load_case so it can be tested against empty dummy files,
    without needing the 4.9 GB dataset on this machine.
    """
    paths = {"CT": find_one(case_dir, "*IMG_CT.nrrd")}
    for name in TARGET_FILES:
        paths[name] = find_one(case_dir, f"*OAR_{name}.seg.nrrd")
    return paths

def load_case(case_dir: str) -> tuple:
    """Read ONE case from disk.

    case_dir : a single case folder, e.g. ".../HaN-Seg/set_1/case_02"
    Returns  : (ct_image, masks_dict) of SimpleITK.Image objects.

    Every mask must sit on the same voxel grid as the CT. If one does not,
    labels would land on the wrong voxels later and nothing would complain,
    so it fails here instead.
    """
    paths = case_files(case_dir)
    ct = sitk.ReadImage(paths["CT"])

    masks = {}
    for name in TARGET_FILES:
        m = sitk.ReadImage(paths[name])
        if m.GetSize() != ct.GetSize() or not np.allclose(m.GetSpacing(), ct.GetSpacing()):
            raise ValueError(
                f"{os.path.basename(case_dir)}/{name}: geometry differs from the CT. "
                f"mask {m.GetSize()} @ {m.GetSpacing()} vs CT {ct.GetSize()} @ {ct.GetSpacing()}"
            )
        masks[name] = m

    return ct, masks


def resample(img, new_spacing=(1.0, 1.0, 1.0), is_mask: bool = False):
    """Put an image on a new, uniform voxel grid.

    is_mask=False : linear interpolation, empty space filled with -1024 (air)
    is_mask=True  : nearest neighbour, empty space filled with 0

    A mask holds class ids, not intensities. Linear interpolation would invent
    0.4 between a 0 and a 1, and rounding that back can erase a structure only
    a few voxels wide. Passing is_mask wrongly is the classic silent way to
    lose the cochlea.
    """
    old_spacing = np.array(img.GetSpacing(), dtype=float)
    old_size = np.array(img.GetSize(), dtype=float)
    new_size = np.round(old_size * old_spacing / np.array(new_spacing, dtype=float))

    # CT scanner saves information in boxes (0.5576 x 0.5576 x 2 mm) which is not a cube. Here the boxes are resampled to cubes (1x1x1mm) so that the vooxel spacing is isotropic.

    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing([float(s) for s in new_spacing])
    r.SetSize([int(s) for s in new_size])
    r.SetOutputOrigin(img.GetOrigin()) # places cube where the original box was
    r.SetOutputDirection(img.GetDirection()) # orientation of the axis of the new cube matches the original box axis
    r.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    r.SetDefaultPixelValue(0 if is_mask else -1024)
    return r.Execute(img)

def window(hu, level: float, width: float):
    """Hounsfield units -> [0, 1], the way a radiologist sets a display window.

    At or below level - width/2 becomes 0, at or above level + width/2 becomes 1,
    everything between is stretched linearly.
      soft tissue : level=40,  width=400
      bone        : level=400, width=1800
    """
    lo = level - width / 2.0
    hi = level + width / 2.0
    return np.clip((np.asarray(hu, dtype=np.float32) - lo) / (hi - lo), 0.0, 1.0)

    # INSERT fig2_two_windows.png INTO SLIDES FOR COMPARISON ON BONE VS SOFT TISSUE

def head_centre(hu: np.ndarray) -> tuple:
    """(row, col) of the patient's centre, from everything denser than air.

    Takes the max over slices first, so one odd slice cannot move the centre.
    Returns the middle of the image if the volume is all air.
    """
    body = hu.max(axis=0) > -500
    ys, xs = np.where(body) # rows and cols
    if len(ys) == 0:
        return hu.shape[1] // 2, hu.shape[2] // 2
    return int((ys.min() + ys.max()) / 2), int((xs.min() + xs.max()) / 2) # returns center of the head in the image


def crop_to(arr: np.ndarray, cy: int, cx: int, size: int = CROP, pad_value=0) -> np.ndarray:
    """Cut a size x size box centred on (cy, cx) out of every slice.
    Pads with pad_value wherever the box runs off the edge of the image."""
    out = np.full((arr.shape[0], size, size), pad_value, dtype=arr.dtype)
    y0, x0 = cy - size // 2, cx - size // 2
    ys, xs = max(y0, 0), max(x0, 0)
    ye, xe = min(y0 + size, arr.shape[1]), min(x0 + size, arr.shape[2])
    out[:, ys - y0:ye - y0, xs - x0:xe - x0] = arr[:, ys:ye, xs:xe]
    return out


def build_label(mask_arrays: dict) -> np.ndarray:
    """Merge the four binary masks into one label volume, 0 background, 1..4.

    Raises if two structures claim the same voxel: one would silently overwrite
    the other and the network would be trained against a label that is wrong.
    """
    shapes = {a.shape for a in mask_arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"masks have different shapes: {shapes}")
    lab = np.zeros(next(iter(mask_arrays.values())).shape, dtype=np.uint8)
    for i, name in enumerate(TARGET_FILES, start=1):
        m = mask_arrays[name] > 0
        clash = int((m & (lab > 0)).sum())
        if clash:
            raise ValueError(f"{name} overlaps an earlier structure in {clash} voxels")
        lab[m] = i
    return lab


def select_slices(lab: np.ndarray, negatives_per_positive: int = 1,
                  seed: int = 0) -> np.ndarray:
    """Slice indices to keep: every slice holding a structure, plus a random
    sample of empty ones.

    Most slices of a head-and-neck CT contain none of the four structures, so
    training on all of them spends nearly all the compute on background.

    This makes the training distribution differ from deployment, where every
    slice is seen. That difference is discussed in the report, not hidden.
    """
    pos = np.where(lab.reshape(len(lab), -1).any(axis=1))[0]
    if len(pos) == 0:
        return pos
    rng = np.random.default_rng(seed)
    pool = np.setdiff1d(np.arange(len(lab)), pos)
    n_neg = min(len(pool), negatives_per_positive * len(pos))
    neg = rng.choice(pool, size=n_neg, replace=False) if n_neg else np.array([], dtype=int)
    return np.sort(np.concatenate([pos, neg]))

def preprocess_case(case_dir: str,
                    spacing=(1.0, 1.0, 1.0),
                    crop: int = CROP,
                    negatives_per_positive: int = 1,
                    seed: int = 0,
                    keep_all: bool = False) -> tuple:
    """One case, from disk to arrays the network can eat.

    Always returns three things, whatever keep_all is set to:
      img   uint8 (n, 2, crop, crop)  ch 0 soft-tissue window, ch 1 bone window
      lab   uint8 (n, crop, crop)     0 background, 1..4 in TARGET_FILES order
      keep  int   (n,)                which resampled slice each row came from

    keep_all=False  every slice holding a structure, plus an equal number of
                    empty ones. For the training patients: it stops the model
                    spending nearly all its compute on background.
    keep_all=True   every slice, so keep is simply 0..Z-1. For the validation
                    and test patients, so they are scored on the slice mix the
                    model would really meet rather than a flattering one.

    Hyperparameters chosen here, all of which the report has to justify:
    1 mm isotropic spacing (so a millimetre means the same in every direction
    when distances are reported later), a 256 mm crop (the smallest box that
    holds both parotids and both cochleae), the two window settings, and
    negatives_per_positive.
    """
    paths = case_files(case_dir)               # one directory listing, not five

    ct = resample(sitk.ReadImage(paths["CT"]), spacing, is_mask=False)
    hu = sitk.GetArrayFromImage(ct)            # (Z, Y, X) int16, Hounsfield units
    del ct

    masks = {}
    for name in TARGET_FILES:
        m = resample(sitk.ReadImage(paths[name]), spacing, is_mask=True)
        masks[name] = sitk.GetArrayFromImage(m) > 0
        del m
    if any(a.shape != hu.shape for a in masks.values()):
        raise ValueError(f"{os.path.basename(case_dir)}: resampled mask does not match CT")

    lab_full = build_label(masks)
    del masks
    before = int((lab_full > 0).sum())

    cy, cx = head_centre(hu)
    hu_c = crop_to(hu, cy, cx, crop, pad_value=-1024)
    lab_c = crop_to(lab_full, cy, cx, crop, pad_value=0)
    del hu, lab_full

    after = int((lab_c > 0).sum())
    if after < before:
        warnings.warn(f"{os.path.basename(case_dir)}: crop lost "
                      f"{before - after} of {before} label voxels")

    soft = (window(hu_c, 40, 400) * 255).astype(np.uint8)
    bone = (window(hu_c, 400, 1800) * 255).astype(np.uint8)
    img = np.stack([soft, bone], axis=1)       # (Z, 2, crop, crop)
    del soft, bone, hu_c

    if keep_all:
        keep = np.arange(len(lab_c))
    else:
        keep = select_slices(lab_c, negatives_per_positive, seed)
    return img[keep], lab_c[keep], keep
