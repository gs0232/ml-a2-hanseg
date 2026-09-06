"""Tests for the preprocessing helpers in src/data.py.

All of these run on synthetic numpy arrays, so they work on the laptop with
no dataset present. They cover the two things that would otherwise fail
silently: a crop that runs off the edge of the image, and two structures
claiming the same voxel.
"""
import numpy as np
import pytest

from src.data import (TARGET_FILES, build_label, crop_to, head_centre,
                      select_slices, window)


# ---------------------------------------------------------------- crop_to

def test_crop_handles_head_near_border():
    """A head jammed into a corner must still give (n, 256, 256), padded with
    air, not an IndexError and not a smaller array."""
    arr = np.full((5, 300, 300), -1024, np.int16)
    arr[:, 10:40, 260:295] = 40
    cy, cx = head_centre(arr)
    out = crop_to(arr, cy, cx, 256, pad_value=-1024)
    assert out.shape == (5, 256, 256)
    assert (out > -500).sum() == (arr > -500).sum()   # no tissue lost
    assert (out == -1024).any()                       # and it really did pad


def test_crop_preserves_everything_when_centred():
    arr = np.zeros((3, 400, 400), np.int16)
    arr[:, 190:210, 190:210] = 1
    out = crop_to(arr, 200, 200, 256)
    assert out.shape == (3, 256, 256)
    assert out.sum() == arr.sum()


def test_crop_keeps_the_dtype():
    arr = np.zeros((2, 300, 300), np.uint8)
    assert crop_to(arr, 150, 150, 256).dtype == np.uint8


# ------------------------------------------------------------ head_centre

def test_head_centre_finds_an_offset_head():
    arr = np.full((4, 512, 512), -1024, np.int16)
    yy, xx = np.ogrid[:512, :512]
    arr[:, (yy - 180) ** 2 + (xx - 300) ** 2 < 95 ** 2] = 40
    assert head_centre(arr) == (180, 300)


def test_head_centre_survives_an_all_air_volume():
    """No patient in the volume must not crash — return the image centre."""
    arr = np.full((2, 64, 64), -1024, np.int16)
    assert head_centre(arr) == (32, 32)


# ------------------------------------------------------------ build_label

def test_build_label_assigns_ids_in_target_order():
    masks = {n: np.zeros((3, 8, 8), bool) for n in TARGET_FILES}
    for i, n in enumerate(TARGET_FILES):
        masks[n][1, i, 0] = True
    lab = build_label(masks)
    assert lab.dtype == np.uint8
    assert sorted(np.unique(lab)) == [0, 1, 2, 3, 4]
    for i, n in enumerate(TARGET_FILES):
        assert lab[1, i, 0] == i + 1


def test_build_label_raises_on_overlap():
    """Two structures on one voxel means one silently overwrote the other."""
    masks = {n: np.zeros((2, 5, 5), bool) for n in TARGET_FILES}
    masks["Cochlea_L"][0, 2, 2] = True
    masks["Parotid_L"][0, 2, 2] = True
    with pytest.raises(ValueError) as e:
        build_label(masks)
    assert "Parotid_L" in str(e.value) and "overlap" in str(e.value)


def test_build_label_raises_on_mismatched_shapes():
    masks = {n: np.zeros((2, 5, 5), bool) for n in TARGET_FILES}
    masks["Cochlea_R"] = np.zeros((2, 6, 6), bool)
    with pytest.raises(ValueError):
        build_label(masks)


# ---------------------------------------------------------- select_slices

def test_select_slices_keeps_every_positive():
    lab = np.zeros((20, 4, 4), np.uint8)
    lab[[3, 4, 11]] = 1
    keep = select_slices(lab, negatives_per_positive=1, seed=0)
    assert {3, 4, 11}.issubset(set(keep.tolist()))
    assert len(keep) == 6                      # 3 positive + 3 negative
    assert list(keep) == sorted(keep)          # indices stay in order


def test_select_slices_is_deterministic():
    lab = np.zeros((40, 4, 4), np.uint8)
    lab[[5, 6, 7]] = 2
    a = select_slices(lab, 1, seed=0)
    assert np.array_equal(a, select_slices(lab, 1, seed=0))
    assert not np.array_equal(a, select_slices(lab, 1, seed=1))


def test_select_slices_handles_a_case_with_no_structure():
    assert len(select_slices(np.zeros((10, 4, 4), np.uint8))) == 0


def test_select_slices_caps_negatives_at_what_exists():
    lab = np.zeros((5, 2, 2), np.uint8)
    lab[[0, 1, 2, 3]] = 1                      # only one empty slice to draw from
    assert len(select_slices(lab, negatives_per_positive=3, seed=0)) == 5


# ----------------------------------------------------------------- window

def test_window_two_channels_differ():
    sl = np.array([[-1000, 40, 1200, 3000]], np.float32)
    soft, bone = window(sl, 40, 400), window(sl, 400, 1800)
    assert not np.allclose(soft, bone)
    assert soft.min() >= 0 and soft.max() <= 1
