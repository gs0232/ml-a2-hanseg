"""Tests for the split. No dataset needed."""
import json
import random
import warnings

import numpy as np
import pytest

from src.dataset import load_or_make_split, make_split

IDS = [f"case_{i:02d}" for i in range(1, 21)]


def test_split_sizes_and_disjointness():
    s = make_split(IDS, seed=0, n_train=12, n_val=4)
    assert (len(s["train"]), len(s["val"]), len(s["test"])) == (12, 4, 4)
    assert not (set(s["train"]) & set(s["val"]))
    assert not (set(s["train"]) & set(s["test"]))
    assert not (set(s["val"]) & set(s["test"]))
    assert sorted(s["train"] + s["val"] + s["test"]) == IDS


def test_split_is_deterministic_for_a_seed():
    assert make_split(IDS, seed=0) == make_split(IDS, seed=0)
    assert make_split(IDS, seed=0)["test"] != make_split(IDS, seed=1)["test"]


def test_split_does_not_depend_on_input_order():
    shuffled = IDS[:]
    random.Random(7).shuffle(shuffled)
    assert make_split(IDS, seed=0) == make_split(shuffled, seed=0)


def test_split_refuses_to_leave_no_test_cases():
    with pytest.raises(ValueError):
        make_split(IDS[:5], n_train=4, n_val=1)


def test_load_or_make_writes_then_reads_the_same_thing(tmp_path):
    p = str(tmp_path / "split.json")
    first = load_or_make_split(p, IDS, seed=0)
    second = load_or_make_split(p, IDS, seed=0)
    assert first == second
    assert json.load(open(p)) == first


def test_the_file_wins_over_a_different_seed(tmp_path):
    """Once written, the split must not change just because the seed did."""
    p = str(tmp_path / "split.json")
    first = load_or_make_split(p, IDS, seed=0)
    again = load_or_make_split(p, IDS, seed=999)
    assert again == first


def test_warns_when_the_cache_no_longer_matches(tmp_path):
    p = str(tmp_path / "split.json")
    load_or_make_split(p, IDS, seed=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_or_make_split(p, IDS + ["case_21"], seed=0)
    assert any("case_21" in str(c.message) for c in caught)
