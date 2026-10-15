import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import preprocessing as prep
import targets

NATIVE_SCALE = (1.625, 0.40625, 0.40625)
DOWNSAMPLE = (1, 4, 4)


def test_downsampled_scale_is_isotropic_for_default_downsample():
    scale = prep.downsampled_scale(NATIVE_SCALE, DOWNSAMPLE)
    assert scale == (1.625, 1.625, 1.625), scale
    print("test_downsampled_scale_is_isotropic_for_default_downsample: PASS", scale)


def test_downsample_volume_shape():
    window = np.random.rand(3, 8, 32, 32)  # (T, Z, Y, X)
    out = prep.downsample_volume(window, DOWNSAMPLE)
    assert out.shape == (3, 8, 8, 8), out.shape
    print("test_downsample_volume_shape: PASS", out.shape)


def test_downsample_volume_single_frame():
    frame = np.random.rand(8, 32, 32)  # (Z, Y, X), no leading T
    out = prep.downsample_volume(frame, DOWNSAMPLE)
    assert out.shape == (8, 8, 8), out.shape
    print("test_downsample_volume_single_frame: PASS", out.shape)


def test_node_coord_downsample_upsample_roundtrip():
    nodes = pd.DataFrame({"z": [4, 20], "y": [40, 8], "x": [16, 100]})
    down = prep.downsample_node_coords(nodes, DOWNSAMPLE)
    back = prep.upsample_node_coords(down, DOWNSAMPLE)
    # z unchanged (dz=1); y/x recovered up to the /4 *4 rounding (exact here
    # since all test values are multiples of 4)
    assert np.allclose(back["z"], nodes["z"])
    assert np.allclose(back["y"], nodes["y"])
    assert np.allclose(back["x"], nodes["x"])
    print("test_node_coord_downsample_upsample_roundtrip: PASS")


def test_preprocess_window_range_and_shape():
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 4000, size=(2, 8, 32, 32)).astype(np.uint16)
    out = prep.preprocess_window(raw, DOWNSAMPLE)
    assert out.shape == (2, 8, 8, 8), out.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 4.0  # quantile_normalize's default clip range
    print("test_preprocess_window_range_and_shape: PASS", out.shape, (out.min(), out.max()))


def test_integration_with_targets_heatmap_peak_lands_correctly():
    # a node at native voxel (z=4, y=40, x=16) -> downsampled (4, 10, 4)
    # (dz=1 so z unchanged; y: 40/4=10; x: 16/4=4)
    nodes_native = pd.DataFrame({"z": [4], "y": [40], "x": [16]})
    nodes_down = prep.downsample_node_coords(nodes_native, DOWNSAMPLE)
    assert nodes_down.iloc[0].tolist() == [4.0, 10.0, 4.0]

    frame_shape_down = (8, 16, 16)  # matches an 8x64x64 native frame downsampled by (1,4,4)
    scale_down = prep.downsampled_scale(NATIVE_SCALE, DOWNSAMPLE)
    hm = targets.build_heatmap_target(frame_shape_down, nodes_down, scale_down, sigma_um=2.0)

    peak = np.unravel_index(np.argmax(hm), hm.shape)
    assert peak == (4, 10, 4), peak
    # isotropic scale -> spread should now be roughly equal across all 3 axes
    # (unlike the native-space anisotropic case tested in test_targets.py)
    z_extent = (hm[:, 10, 4] > 0.01).sum()
    y_extent = (hm[4, :, 4] > 0.01).sum()
    x_extent = (hm[4, 10, :] > 0.01).sum()
    assert abs(z_extent - y_extent) <= 1, (z_extent, y_extent)
    assert abs(z_extent - x_extent) <= 1, (z_extent, x_extent)
    print("test_integration_with_targets_heatmap_peak_lands_correctly: PASS",
          "peak=", peak, "extents=", (z_extent, y_extent, x_extent))


if __name__ == "__main__":
    test_downsampled_scale_is_isotropic_for_default_downsample()
    test_downsample_volume_shape()
    test_downsample_volume_single_frame()
    test_node_coord_downsample_upsample_roundtrip()
    test_preprocess_window_range_and_shape()
    test_integration_with_targets_heatmap_peak_lands_correctly()
    print("\nALL PASS")
