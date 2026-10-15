import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from detector_model import Detector3D, DetectorConfig
import targets


def _tiny_config():
    # small config for fast tests -- real training uses the default (32,64,128)
    return DetectorConfig(unet_out_channels=8, unet_layers=(8, 16), gradient_checkpointing=False)


def test_forward_shape_and_range():
    model = Detector3D(_tiny_config())
    model.eval()
    window = torch.rand(2, 3, 16, 32, 32)  # (B=2, T=3, Z=16, Y=32, X=32)
    with torch.no_grad():
        out = model(window)
    assert out.shape == (2, 3, 16, 32, 32), out.shape
    assert out.min() >= 0.0 and out.max() <= 1.0, (out.min().item(), out.max().item())
    print("test_forward_shape_and_range: PASS", "shape=", tuple(out.shape),
          "range=", (out.min().item(), out.max().item()))


def test_forward_rejects_wrong_ndim():
    model = Detector3D(_tiny_config())
    bad = torch.rand(2, 16, 32, 32)  # missing T dim
    raised = False
    try:
        model(bad)
    except ValueError:
        raised = True
    assert raised, "expected ValueError on wrong input ndim"
    print("test_forward_rejects_wrong_ndim: PASS")


def test_detect_single_frame_matches_forward_first_frame():
    model = Detector3D(_tiny_config())
    model.eval()
    frame = torch.rand(2, 16, 32, 32)  # (B, Z, Y, X)
    with torch.no_grad():
        single = model.detect_single_frame(frame)
        window = torch.stack([frame, frame], dim=1)
        full = model(window)
    assert single.shape == (2, 16, 32, 32), single.shape
    assert torch.allclose(single, full[:, 0]), "detect_single_frame should match forward()[:,0] on a duplicated frame"
    print("test_detect_single_frame_matches_forward_first_frame: PASS")


def test_gradient_step_reduces_loss_against_real_target():
    # end-to-end: real Gaussian heatmap target from targets.py, real weighted
    # loss, one actual optimizer step -- confirms the whole thing is trainable,
    # not just shape-correct.
    torch.manual_seed(0)
    model = Detector3D(_tiny_config())
    model.train()

    Z, Y, X = 16, 32, 32
    scale = (1.625, 0.40625, 0.40625)
    window = torch.rand(1, 2, Z, Y, X)

    import pandas as pd
    nodes_t = pd.DataFrame({"z": [8, 8], "y": [10, 22], "x": [16, 16]})
    heatmap_np = targets.build_heatmap_target((Z, Y, X), nodes_t, scale, sigma_um=2.0)
    weight_np = targets.build_loss_weight_map((Z, Y, X), nodes_t, scale, background_weight=0.2)
    target = torch.from_numpy(heatmap_np).unsqueeze(0)  # (1, Z, Y, X) -- single frame's worth
    weight = torch.from_numpy(weight_np).unsqueeze(0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    def step():
        optimizer.zero_grad()
        pred = model(window)[:, 0]  # take first frame's prediction, (1, Z, Y, X)
        loss = targets.focal_heatmap_loss(pred, target, weight)
        loss.backward()
        optimizer.step()
        return loss.item()

    losses = [step() for _ in range(15)]
    assert losses[-1] < losses[0], (losses[0], losses[-1])
    print("test_gradient_step_reduces_loss_against_real_target: PASS",
          "loss[0]=", round(losses[0], 5), "loss[-1]=", round(losses[-1], 5))


def test_gradient_checkpointing_path_runs_in_train_mode():
    # gradient_checkpointing=True is the real default -- make sure that
    # code path actually runs (it's skipped by the tiny-config tests above)
    config = DetectorConfig(unet_out_channels=8, unet_layers=(8, 16), gradient_checkpointing=True)
    model = Detector3D(config)
    model.train()
    window = torch.rand(1, 2, 16, 32, 32, requires_grad=False)
    out = model(window)
    loss = out.mean()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "no gradients flowed with gradient_checkpointing=True"
    print("test_gradient_checkpointing_path_runs_in_train_mode: PASS")


if __name__ == "__main__":
    test_forward_shape_and_range()
    test_forward_rejects_wrong_ndim()
    test_detect_single_frame_matches_forward_first_frame()
    test_gradient_step_reduces_loss_against_real_target()
    test_gradient_checkpointing_path_runs_in_train_mode()
    print("\nALL PASS")
