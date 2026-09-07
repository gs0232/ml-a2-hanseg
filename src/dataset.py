"""The patient-level split, and reading the cache back off disk."""
import glob
import json
import os
import warnings

import numpy as np


def cached_case_ids(cache_dir: str) -> list:
    """Sorted ids of every case in the cache, e.g. ['case_01', 'case_02', ...]."""
    return sorted(os.path.basename(f)[:-4]
                  for f in glob.glob(os.path.join(cache_dir, "*.npz")))


def make_split(case_ids, seed: int = 0, n_train: int = 12, n_val: int = 4) -> dict:
    """Split BY PATIENT into train / val / test.

    By patient, not by slice: adjacent slices of the same cochlea are almost
    identical, so a slice-level split would put near-copies of the test data
    in the training set and the reported score would be meaningless. The
    marking rubric names data leakage explicitly.
    """
    ids = sorted(case_ids)
    if n_train + n_val >= len(ids):
        raise ValueError(f"{len(ids)} cases cannot give {n_train} train + "
                         f"{n_val} val and leave any for test")
    order = [ids[i] for i in np.random.default_rng(seed).permutation(len(ids))]
    return {"seed": seed,
            "train": sorted(order[:n_train]),
            "val": sorted(order[n_train:n_train + n_val]),
            "test": sorted(order[n_train + n_val:])}


def load_or_make_split(path: str, case_ids=None, seed: int = 0,
                       n_train: int = 12, n_val: int = 4) -> dict:
    """Read the split from `path`, or create and write it if absent.

    The FILE is the authority, not this function. Once written it is never
    regenerated, because a split that quietly changes when the cache or the
    seed changes makes every earlier number unreproducible. If the cache no
    longer matches the file, that is a warning, not a silent re-split.
    """
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path) as f:
            split = json.load(f)
        if case_ids is not None:
            in_file = set(split["train"]) | set(split["val"]) | set(split["test"])
            missing = in_file - set(case_ids)
            extra = set(case_ids) - in_file
            if missing:
                warnings.warn(f"{path} names cases that are not cached: {sorted(missing)}")
            if extra:
                warnings.warn(f"cached cases missing from {path}: {sorted(extra)} "
                              f"— delete the file to re-split, and redo every result")
        return split

    if case_ids is None:
        raise ValueError(f"{path} does not exist and no case_ids were given")
    split = make_split(case_ids, seed, n_train, n_val)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=2)
    return split


def load_cases(cache_dir: str, case_ids):
    """Concatenate several cached cases into one (img, lab, case_of_row) set.

    case_of_row says which patient each slice came from, so per-case metrics
    can be computed afterwards — the metrics are 3-D per patient, never
    averaged over a pile of loose slices.
    """
    imgs, labs, owner = [], [], []
    for cid in case_ids:
        d = np.load(os.path.join(cache_dir, f"{cid}.npz"))
        imgs.append(d["img"])
        labs.append(d["lab"])
        owner += [cid] * len(d["lab"])
    return np.concatenate(imgs), np.concatenate(labs), np.array(owner)
