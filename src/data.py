import os
import glob
import SimpleITK as sitk
import numpy as np

TARGET_FILES = ["Cochlea_L", "Cochlea_R", "Parotid_L", "Parotid_R"]

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

    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing([float(s) for s in new_spacing])
    r.SetSize([int(s) for s in new_size])
    r.SetOutputOrigin(img.GetOrigin())
    r.SetOutputDirection(img.GetDirection())
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

def preprocess_case(case_dir: str) -> tuple:
    """Returns (img, lab, kept_slice_indices).
    img : uint8 (n, 2, 256, 256)  — channel 0 soft-tissue window, channel 1 bone
    lab : uint8 (n, 256, 256)     — 0 bg, 1 Cochlea_L, 2 Cochlea_R, 3 Parotid_L, 4 Parotid_R
    n   : slices containing any structure, plus an equal number of empty ones"""