def load_case(case_dir: str) -> tuple:
    """Returns (ct_image, masks_dict).
    ct_image      : SimpleITK.Image, the CT volume
    masks_dict    : {"Cochlea_L": Image, "Cochlea_R": ..., "Parotid_L": ..., "Parotid_R": ...}
    Must raise, loudly, if a expected file is missing — never return None silently."""

def resample(img, new_spacing=(1.0, 1.0, 1.0), is_mask: bool = False):
    """Returns a SimpleITK.Image on an isotropic 1 mm grid.
    is_mask=True  -> nearest-neighbour interpolation, default value 0
    is_mask=False -> linear interpolation, default value -1024 (air)
    Getting this backwards silently destroys thin structures."""

def window(hu, level: float, width: float):
    """Hounsfield units -> float array in [0, 1], clipped at the window edges."""

def preprocess_case(case_dir: str) -> tuple:
    """Returns (img, lab, kept_slice_indices).
    img : uint8 (n, 2, 256, 256)  — channel 0 soft-tissue window, channel 1 bone
    lab : uint8 (n, 256, 256)     — 0 bg, 1 Cochlea_L, 2 Cochlea_R, 3 Parotid_L, 4 Parotid_R
    n   : slices containing any structure, plus an equal number of empty ones"""