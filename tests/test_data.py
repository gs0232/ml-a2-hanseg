# tests/test_data.py — write the test before the function. It is faster, and it
# gives you something concrete to point at when asked "how did you verify this?"

def test_crop_handles_head_near_border():
    """A head whose centre is 20 px from the edge must still produce a
    (n, 256, 256) array, padded with air, not an IndexError."""

def test_resample_preserves_cochlea():
    """Voxel count of a cochlea mask after 1 mm resampling should be within
    ~20% of (original count x spacing ratio). If it is near zero, your
    interpolator is wrong."""

def test_labels_are_disjoint():
    """No voxel may carry two structure labels. If Cochlea_L and Cochlea_R
    overlap, your loop is overwriting rather than accumulating."""

def test_dice_known_values():
    """identical masks -> 1.0 ; disjoint -> 0.0 ; quarter overlap -> 0.25 ;
    class absent from both -> NaN, not a crash and not 0.0."""