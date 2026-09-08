"""The training loop, and running a trained model over a whole case."""
import time

import numpy as np
import torch

from src.losses import get_loss


def foreground_dice(pred, target, n_classes: int = 5):
    """Per-class Dice accumulated over a loader, classes 1..4.

    This is a fast SLICE-level score for watching training. The real numbers
    in the report come from src.metrics on reassembled 3-D volumes; a slice
    average is not the same thing and must not be reported as if it were.
    """
    inter = torch.zeros(n_classes, device=pred.device)
    denom = torch.zeros(n_classes, device=pred.device)
    for c in range(n_classes):
        p, g = pred == c, target == c
        inter[c] = (p & g).sum()
        denom[c] = p.sum() + g.sum()
    return inter, denom


@torch.no_grad()
def validate(model, loader, device, n_classes: int = 5):
    model.eval()
    inter = torch.zeros(n_classes, device=device)
    denom = torch.zeros(n_classes, device=device)
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        i, d = foreground_dice(pred, y, n_classes)
        inter += i
        denom += d
    dice = torch.where(denom > 0, 2 * inter / denom.clamp(min=1),
                       torch.full_like(denom, float("nan")))
    return dice.cpu().numpy()


def fit(model, train_loader, val_loader, device, loss_name: str = "compound",
        epochs: int = 30, lr: float = 1e-3, weight_decay: float = 1e-4,
        class_names=("Cochlea_L", "Cochlea_R", "Parotid_L", "Parotid_R"),
        log_every: int = 1, amp: bool = True):
    """Train, and return (history, best_state_dict).

    "Best" is the highest mean Dice over the four structures on the validation
    patients — not the lowest training loss, and not accuracy. Chosen on
    validation and never on test, so the test number stays honest.

    amp=True runs the forward pass in 16-bit where that is safe, which roughly
    halves the time on a T4. The GradScaler is not optional with it: 16-bit
    gradients for a structure this small underflow to zero, and the scaler
    multiplies the loss up before the backward pass and divides back after.
    """
    loss_fn = get_loss(loss_name)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best_score, best_state = [], -1.0, None
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        running, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()

        dice = validate(model, val_loader, device)
        score = float(np.nanmean(dice[1:]))
        history.append({"epoch": epoch, "train_loss": running / max(n, 1),
                        "val_mean_dice": score,
                        **{f"val_dice_{c}": float(dice[i])
                           for i, c in enumerate(class_names, start=1)}})
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:3d}  loss {running/max(n,1):.4f}  " +
                  "  ".join(f"{c[:7]} {dice[i]:.3f}"
                            for i, c in enumerate(class_names, start=1)) +
                  f"   mean {score:.3f}  [{time.time()-t0:.0f}s]")
    print(f"best mean validation Dice {best_score:.4f}")
    return history, best_state


@torch.no_grad()
def predict_case(model, cache_dir: str, case_id: str, device,
                 batch_size: int = 8) -> tuple:
    """Run the model over every slice of one case and reassemble the volume.

    Returns (pred_lab, gt_lab), both (Z, H, W) uint8 — the 3-D arrays that
    src.metrics needs. Every slice, not the selected ones: at deployment the
    model sees the whole scan.
    """
    model.eval()
    d = np.load(f"{cache_dir}/{case_id}.npz")
    img, gt = d["img"], d["lab"]
    out = np.zeros_like(gt)
    for i in range(0, len(img), batch_size):
        x = torch.from_numpy(img[i:i + batch_size].astype(np.float32) / 255.0).to(device)
        out[i:i + batch_size] = model(x).argmax(1).cpu().numpy().astype(np.uint8)
    return out, gt
