"""Tests for src/metrics.py. Pure numpy, no dataset needed."""
import numpy as np
import pytest

from src.metrics import (dice, evaluate_case, hd, hd95, hd_max, surface,
                         surface_dice, surface_distances)

NAMES = ["Cochlea_L", "Cochlea_R", "Parotid_L", "Parotid_R"]


def block(shape=(30, 30, 30), lo=(10, 10, 10), hi=(20, 20, 20)):
    m = np.zeros(shape, bool)
    m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    return m


# ------------------------------------------------------------------- dice

def test_dice_identical_is_one():
    m = block()
    assert dice(m, m) == 1.0


def test_dice_disjoint_is_zero():
    a = block(lo=(2, 2, 2), hi=(6, 6, 6))
    b = block(lo=(20, 20, 20), hi=(24, 24, 24))
    assert dice(a, b) == 0.0


def test_dice_both_empty_is_nan():
    z = np.zeros((5, 5, 5), bool)
    assert np.isnan(dice(z, z))


def test_dice_quarter_overlap():
    """2x2 vs 2x2 sharing one cell -> 2*1/(4+4) = 0.25"""
    a = np.zeros((1, 4, 4), bool); a[0, 1:3, 1:3] = True
    b = np.zeros((1, 4, 4), bool); b[0, 2:4, 2:4] = True
    assert dice(a, b) == pytest.approx(0.25)


def test_dice_is_symmetric():
    a, b = block(), block(lo=(12, 12, 12), hi=(22, 22, 22))
    assert dice(a, b) == dice(b, a)


# ---------------------------------------------------------------- surface

def test_surface_of_a_block_is_hollow():
    m = block()
    s = surface(m)
    assert s.sum() < m.sum()
    assert not s[15, 15, 15]        # the middle is interior, not surface
    assert s[10, 15, 15]            # the face is surface


def test_surface_of_empty_is_empty():
    assert not surface(np.zeros((4, 4, 4), bool)).any()


# --------------------------------------------------------------------- hd

def test_hd95_identical_is_zero():
    m = block()
    assert hd95(m, m) == 0.0


def test_hd95_grows_with_the_shift():
    gt = block()
    close = hd95(gt, block(lo=(11, 10, 10), hi=(21, 20, 20)))
    far = hd95(gt, block(lo=(15, 10, 10), hi=(25, 20, 20)))
    assert 0 < close < far


def test_hd95_is_nan_when_prediction_is_empty():
    assert np.isnan(hd95(np.zeros((10, 10, 10), bool), block()))


def test_hd95_respects_spacing():
    """Same voxel shift, but 2 mm voxels along z -> twice the distance."""
    gt, pred = block(), block(lo=(13, 10, 10), hi=(23, 20, 20))
    one = hd95(gt, pred, spacing=(1.0, 1.0, 1.0))
    two = hd95(gt, pred, spacing=(2.0, 1.0, 1.0))
    assert two == pytest.approx(2 * one, abs=0.5)


def test_hd_max_is_at_least_hd95():
    gt, pred = block(), block(lo=(13, 10, 10), hi=(23, 20, 20))
    assert hd_max(pred, gt) >= hd95(pred, gt)


# ----------------------------------------------------------- surface_dice

def test_surface_dice_identical_is_one():
    m = block()
    assert surface_dice(m, m, tol_mm=0.5) == 1.0


def test_surface_dice_rises_with_tolerance():
    gt, pred = block(), block(lo=(13, 10, 10), hi=(23, 20, 20))
    scores = [surface_dice(gt, pred, tol_mm=t) for t in (0.5, 1.5, 3.0, 10.0)]
    assert scores == sorted(scores)
    assert scores[-1] == pytest.approx(1.0)


def test_surface_dice_is_nan_when_prediction_is_empty():
    assert np.isnan(surface_dice(np.zeros((8, 8, 8), bool), block()))


# ------------------- the claim the whole report rests on ------------------

def test_each_metric_is_blind_to_something_different():
    """One prediction is uniformly 1 mm off. The other fits well but sends a
    thin spur 12 mm into the wrong place. Measured behaviour:

        metric        uniform shift    thin spur
        dice              0.900          0.872     barely notices
        hd95              1.00           1.41      barely notices
        hd_max            1.00          12.00      sees it
        sdice @ 1 mm      1.000          0.943     sees it, mildly

    hd95 misses it because the spur is only ~9% of the boundary and the 95th
    percentile throws away the worst 5%. So no single number is enough, and
    the report has to give more than one. If this test fails, that argument
    has changed and the discussion section needs rewriting.
    """
    gt = block(shape=(40, 40, 40), lo=(15, 15, 15), hi=(25, 25, 25))
    shifted = block(shape=(40, 40, 40), lo=(16, 15, 15), hi=(26, 25, 25))
    spur = block(shape=(40, 40, 40), lo=(16, 16, 15), hi=(25, 25, 25))
    spur[3:15, 19:21, 19:21] = True

    assert abs(dice(shifted, gt) - dice(spur, gt)) < 0.05
    assert abs(hd95(shifted, gt) - hd95(spur, gt)) < 1.0
    assert hd_max(spur, gt) > 5 * hd_max(shifted, gt)
    assert surface_dice(spur, gt, tol_mm=1.0) < surface_dice(shifted, gt, tol_mm=1.0)


# ---------------------------------------------------------- evaluate_case

def test_evaluate_case_returns_one_row_per_structure():
    gt = np.zeros((20, 20, 20), np.uint8)
    gt[5, 5, 5] = 1
    gt[6, 6:9, 6:9] = 3
    rows = evaluate_case(gt.copy(), gt, NAMES)
    assert len(rows) == 4
    assert [r["structure"] for r in rows] == NAMES
    assert rows[0]["dice"] == 1.0
    assert np.isnan(rows[1]["dice"])          # Cochlea_R absent from both
    assert rows[2]["dice"] == 1.0


def test_evaluate_case_flags_a_complete_miss():
    gt = np.zeros((20, 20, 20), np.uint8)
    gt[6, 6:9, 6:9] = 1
    rows = evaluate_case(np.zeros_like(gt), gt, NAMES)
    assert rows[0]["missed"] is True
    assert rows[0]["dice"] == 0.0
    assert np.isnan(rows[0]["hd95_mm"])
    assert rows[1]["missed"] is False         # absent from both, not a miss


def test_evaluate_case_flags_a_false_positive_on_an_absent_structure():
    gt = np.zeros((20, 20, 20), np.uint8)
    pred = np.zeros_like(gt)
    pred[6, 6:9, 6:9] = 2
    rows = evaluate_case(pred, gt, NAMES)
    assert rows[1]["false_positive_only"] is True
    assert rows[1]["dice"] == 0.0
