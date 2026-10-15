import os
import sys

import torch

import data_io as bio_io
from detector_model import Detector3D, DetectorConfig
import train_detector as td

ROOT = "/kaggle/input/competitions/biohub-cell-tracking-during-development/train"

if not os.path.isdir(ROOT):
    import make_fixtures
    make_fixtures.main()


def test_build_training_window_shapes():
    samples = bio_io.list_samples(ROOT)
    s = samples[0]

    # Inspect the actual native volume dimensions rather than hardcoding
    # them -- ties this test to whatever the fixture actually is, so it
    # keeps working if the fixture shape ever changes.
    vol = bio_io.open_zarr_volume(s.zarr_path)
    native_shape = vol.shape  # (T, Z, Y, X)

    window_size = 2
    downsample = td.DOWNSAMPLE  # (1, 4, 4)

    window, target, weight = td.build_training_window(
        s, t_start=0, window_size=window_size,
    )

    # Expected spatial dimensions after downsampling -- computed from the
    # real native shape, not a separately-hardcoded literal.
    expected_z = native_shape[1] // downsample[0]
    expected_y = native_shape[2] // downsample[1]
    expected_x = native_shape[3] // downsample[2]

    expected_shape = (
        1,              # batch dimension
        window_size,    # temporal window
        expected_z,
        expected_y,
        expected_x,
    )

    assert window.shape == expected_shape, (
        f"Expected window shape {expected_shape}, got {tuple(window.shape)}"
    )
    assert target.shape == window.shape, (
        f"Expected target shape {tuple(window.shape)}, got {tuple(target.shape)}"
    )
    assert weight.shape == window.shape, (
        f"Expected weight shape {tuple(window.shape)}, got {tuple(weight.shape)}"
    )
    assert window.dtype == torch.float32

    print(
        "test_build_training_window_shapes: PASS",
        f"native={native_shape}", f"downsample={downsample}", f"output={tuple(window.shape)}",
    )


def test_train_step_runs_and_reduces_loss_on_real_fixture():
    samples = bio_io.list_samples(ROOT)
    s = samples[0]

    config = DetectorConfig(unet_out_channels=8, unet_layers=(8, 16), gradient_checkpointing=False)
    model = Detector3D(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    losses = [td.train_step(model, optimizer, s, t_start=0, window_size=2) for _ in range(15)]
    assert losses[-1] < losses[0], (losses[0], losses[-1])
    print("test_train_step_runs_and_reduces_loss_on_real_fixture: PASS",
          "loss[0]=", round(losses[0], 5), "loss[-1]=", round(losses[-1], 5))


def test_train_step_raises_on_no_geff():
    samples = bio_io.list_samples(ROOT)
    s = samples[0]
    # simulate a test-set sample (no ground truth) by clearing geff_path
    from dataclasses import replace
    test_sample = replace(s, geff_path=None)

    config = DetectorConfig(unet_out_channels=8, unet_layers=(8, 16), gradient_checkpointing=False)
    model = Detector3D(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    raised = False
    try:
        td.train_step(model, optimizer, test_sample, t_start=0, window_size=2)
    except ValueError:
        raised = True
    assert raised, "expected ValueError when training on a sample with no ground truth"
    print("test_train_step_raises_on_no_geff: PASS")


def test_build_training_window_respects_device_param():
    samples = bio_io.list_samples(ROOT)
    s = samples[0]
    window, target, weight = td.build_training_window(s, t_start=0, window_size=2, device="cpu")
    assert window.device.type == "cpu"
    assert target.device.type == "cpu"
    assert weight.device.type == "cpu"
    print("test_build_training_window_respects_device_param: PASS")


def test_train_step_respects_device_param():
    samples = bio_io.list_samples(ROOT)
    s = samples[0]
    config = DetectorConfig(unet_out_channels=8, unet_layers=(8, 16), gradient_checkpointing=False)
    model = Detector3D(config).to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    # explicit device="cpu" -- confirms train_step actually threads the
    # device argument through to build_training_window rather than
    # silently defaulting and relying on model/data happening to agree
    loss = td.train_step(model, optimizer, s, t_start=0, window_size=2, device="cpu")
    assert isinstance(loss, float)
    print("test_train_step_respects_device_param: PASS")


if __name__ == "__main__":
    test_build_training_window_shapes()
    test_build_training_window_respects_device_param()
    test_train_step_respects_device_param()
    test_train_step_runs_and_reduces_loss_on_real_fixture()
    test_train_step_raises_on_no_geff()
    print("\nALL PASS")