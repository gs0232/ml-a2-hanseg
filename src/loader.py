"""Turning the cached .npz files into something PyTorch can iterate.

Kept separate from dataset.py so that dataset.py stays pure numpy and its
tests still run on a laptop with no torch installed.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data import select_slices

# A left-right flip moves each structure to the other side of the head, so the
# LABELS have to be swapped as well: what was the left cochlea is now on the
# right. Flipping without this teaches the network that laterality is
# arbitrary, and it then cannot tell Cochlea_L from Cochlea_R at all.
#            bg  Coch_L Coch_R Par_L Par_R
FLIP_SWAP = [0,   2,     1,     4,    3]


class SliceDataset(Dataset):
    """2-D slices from a set of cached cases.

    select=True  applies select_slices: every slice holding a structure plus
                 an equal number of empty ones. Use for TRAINING — the cache
                 holds every slice, and 85% of them are empty.
    select=False keeps every slice. Use for validation and test, so the score
                 reflects the slice mix the model would really meet.
    """

    def __init__(self, cache_dir: str, case_ids, select: bool = True,
                 negatives_per_positive: int = 1, seed: int = 0,
                 augment: bool = False):
        self.augment = augment
        self.items, self.index, self.case_of = [], [], []
        for c, cid in enumerate(sorted(case_ids)):
            d = np.load(f"{cache_dir}/{cid}.npz")
            img, lab = d["img"], d["lab"]
            keep = (select_slices(lab, negatives_per_positive, seed)
                    if select else np.arange(len(lab)))
            self.items.append((img, lab))
            self.index += [(c, int(s)) for s in keep]
            self.case_of += [cid] * len(keep)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        c, s = self.index[i]
        img, lab = self.items[c]
        x = torch.from_numpy(img[s].astype(np.float32) / 255.0)   # (2, H, W)
        y = torch.from_numpy(lab[s].astype(np.int64))             # (H, W)
        if self.augment and torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[-1])
            y = torch.as_tensor(FLIP_SWAP, dtype=torch.int64)[torch.flip(y, dims=[-1])]
        return x, y

    def class_counts(self) -> np.ndarray:
        """Voxels per class over everything this dataset will serve.
        Useful for sanity-checking the imbalance, and for class weights."""
        counts = np.zeros(5, dtype=np.int64)
        for c, s in self.index:
            counts += np.bincount(self.items[c][1][s].ravel(), minlength=5)
        return counts


def make_loaders(cache_dir: str, split: dict, batch_size: int = 8,
                 negatives_per_positive: int = 1, seed: int = 0,
                 augment: bool = True, num_workers: int = 2):
    """(train_loader, val_loader). Validation keeps every slice deliberately."""
    train_ds = SliceDataset(cache_dir, split["train"], select=True,
                            negatives_per_positive=negatives_per_positive,
                            seed=seed, augment=augment)
    val_ds = SliceDataset(cache_dir, split["val"], select=False)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, drop_last=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers))
