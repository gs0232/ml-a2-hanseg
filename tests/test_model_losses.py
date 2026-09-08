"""Tests for the network and the losses.

These need torch, which is on Colab but not on the laptop, so the whole file
skips cleanly when torch is missing. Run it in Colab after cloning.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.losses import compound_loss, dice_loss, get_loss, tversky_loss
from src.model import UNet2D


# ------------------------------------------------------------------ model

def test_output_shape_matches_input():
    m = UNet2D(in_ch=2, n_classes=5)
    out = m(torch.zeros(2, 2, 256, 256))
    assert out.shape == (2, 5, 256, 256)


def test_parameter_count_is_what_the_docstring_claims():
    n = UNet2D().n_parameters()
    assert n == 7_762_885, (
        f"got {n:,} parameters, expected 7,762,885. If you changed `base` or "
        f"the block, update the docstring and this number together.")


def test_output_is_logits_not_probabilities():
    """If the head ever gains a softmax, the losses would apply a second one."""
    m = UNet2D()
    out = m(torch.randn(1, 2, 64, 64))
    sums = out.softmax(1).sum(1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    assert not torch.allclose(out.sum(1), torch.ones_like(out.sum(1)), atol=1e-3)


def test_gradients_reach_the_first_layer():
    """A broken skip connection or a detached tensor shows up here."""
    m = UNet2D()
    loss = dice_loss(m(torch.randn(1, 2, 64, 64)), torch.zeros(1, 64, 64, dtype=torch.long))
    loss.backward()
    g = m.enc1[0].weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_accepts_any_size_divisible_by_16():
    m = UNet2D()
    for s in (64, 128, 256):
        assert m(torch.zeros(1, 2, s, s)).shape == (1, 5, s, s)


# ----------------------------------------------------------------- losses

def perfect_logits(y, n_classes=5, scale=20.0):
    return torch.nn.functional.one_hot(y, n_classes).permute(0, 3, 1, 2).float() * scale


def make_target():
    y = torch.zeros(2, 32, 32, dtype=torch.long)
    y[:, 8:16, 8:16] = 1
    y[:, 20:30, 20:30] = 3
    return y


def test_dice_loss_is_near_zero_for_a_perfect_prediction():
    y = make_target()
    assert dice_loss(perfect_logits(y), y).item() < 1e-3


def test_dice_loss_is_high_for_a_blank_prediction():
    y = make_target()
    blank = torch.zeros(2, 5, 32, 32)
    blank[:, 0] = 10.0
    assert dice_loss(blank, y).item() > 0.4


def test_dice_is_exactly_tversky_at_half_half():
    y = make_target()
    lg = torch.randn(2, 5, 32, 32)
    assert torch.allclose(dice_loss(lg, y), tversky_loss(lg, y, 0.5, 0.5), atol=0)


def test_eps_of_order_one_forgives_a_correctly_absent_class():
    """With eps below softmax's noise floor, a PERFECT prediction scores badly.
    Measured in numpy beforehand: loss 0.75 at eps=1e-8 versus 0.0002 at
    eps=1.0. This is why eps is a scale choice, not a zero guard."""
    y = torch.zeros(2, 64, 64, dtype=torch.long)
    y[:, 20:28, 20:28] = 1
    lg = perfect_logits(y)
    assert dice_loss(lg, y, eps=1.0).item() < 0.01
    assert dice_loss(lg, y, eps=1e-8).item() > 0.5


def test_alpha_above_beta_punishes_under_segmentation_more():
    """The asymmetry the surgical framing needs."""
    y = torch.zeros(1, 32, 32, dtype=torch.long)
    y[:, 10:22, 10:22] = 1
    under = torch.zeros(1, 5, 32, 32); under[:, 0] = 6; under[:, 1, 14:18, 14:18] = 12
    over = torch.zeros(1, 5, 32, 32); over[:, 0] = 6; over[:, 1, 6:26, 6:26] = 12
    gap_sym = tversky_loss(under, y, 0.5, 0.5) - tversky_loss(over, y, 0.5, 0.5)
    gap_asym = tversky_loss(under, y, 0.7, 0.3) - tversky_loss(over, y, 0.7, 0.3)
    assert gap_asym > gap_sym


def test_background_is_excluded_from_the_mean():
    """Including background would let a blank prediction look good, because
    background is 99.8% of the voxels."""
    y = make_target()
    blank = torch.zeros(2, 5, 32, 32); blank[:, 0] = 10.0
    assert dice_loss(blank, y, ignore_background=False).item() < \
           dice_loss(blank, y, ignore_background=True).item()


def test_every_loss_in_the_factory_runs_and_is_finite():
    y = make_target()
    lg = torch.randn(2, 5, 32, 32, requires_grad=True)
    for name in ("ce", "dice", "compound", "tversky"):
        v = get_loss(name)(lg, y)
        assert torch.isfinite(v), name
        v.backward(retain_graph=True)


def test_unknown_loss_name_raises():
    with pytest.raises(ValueError):
        get_loss("nonsense")


# ------------------------------------------- the augmentation label swap

def test_flip_swaps_left_and_right_labels():
    """Flipping the image without swapping the labels would teach the network
    that laterality is arbitrary."""
    from src.loader import FLIP_SWAP
    assert FLIP_SWAP == [0, 2, 1, 4, 3]
    y = torch.tensor([[[0, 1], [3, 0]]])
    flipped = torch.as_tensor(FLIP_SWAP)[torch.flip(y, dims=[-1])]
    assert flipped.tolist() == [[[2, 0], [0, 4]]]
