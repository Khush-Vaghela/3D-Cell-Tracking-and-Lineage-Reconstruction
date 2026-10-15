import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import detect
import preprocessing as prep
import data_io as bio_io
from detector_model import Detector3D, DetectorConfig
from pipeline_config import DOWNSAMPLE

ROOT = "/kaggle/input/competitions/biohub-cell-tracking-during-development/train"

if not os.path.isdir(ROOT):
    import make_fixtures
    make_fixtures.main()

# The production scale calculation, not a hardcoded duplicate of it -- a
# hardcoded (1.625, 1.625, 1.625) here would silently stop testing the real
# pipeline the moment NATIVE_SCALE or DOWNSAMPLE changed elsewhere.
SCALE_DOWN = prep.downsampled_scale(bio_io.NATIVE_SCALE, DOWNSAMPLE)


def _synthetic_heatmap(shape, peaks):
    """peaks: list of (z, y, x, value)."""
    hm = np.zeros(shape, dtype=np.float32)
    for z, y, x, v in peaks:
        hm[z, y, x] = v
    return hm


def test_extract_peaks_finds_isolated_peak():
    hm = _synthetic_heatmap((16, 16, 16), [(8, 8, 8, 0.9)])
    peaks = detect.extract_peaks_single_frame(hm, threshold=0.3, min_distance_um=5.0, scale_down=SCALE_DOWN)
    assert len(peaks) == 1
    assert peaks.iloc[0][["z", "y", "x"]].tolist() == [8, 8, 8]
    assert np.isclose(peaks.iloc[0]["score"], 0.9)  # float32 storage -- exact == is fragile
    print("test_extract_peaks_finds_isolated_peak: PASS")


def test_extract_peaks_below_threshold_dropped():
    hm = _synthetic_heatmap((16, 16, 16), [(8, 8, 8, 0.9), (2, 2, 2, 0.1)])
    peaks = detect.extract_peaks_single_frame(hm, threshold=0.3, min_distance_um=5.0, scale_down=SCALE_DOWN)
    assert len(peaks) == 1  # the 0.1 peak is below threshold
    print("test_extract_peaks_below_threshold_dropped: PASS")


def test_extract_peaks_nms_suppresses_nearby_weaker_peak():
    # two peaks 2 voxels apart -- physical distance depends on SCALE_DOWN,
    # not hardcoded, but at the real (1,4,4)/(1.625,0.40625,0.40625)
    # combination that's 2*1.625=3.25um, comfortably inside a 10um radius
    hm = _synthetic_heatmap((16, 16, 16), [(8, 8, 8, 0.9), (8, 8, 10, 0.6)])
    peaks = detect.extract_peaks_single_frame(hm, threshold=0.3, min_distance_um=10.0, scale_down=SCALE_DOWN)
    assert len(peaks) == 1
    assert np.isclose(peaks.iloc[0]["score"], 0.9)  # the stronger one survives
    print("test_extract_peaks_nms_suppresses_nearby_weaker_peak: PASS")


def test_extract_peaks_far_apart_both_kept():
    hm = _synthetic_heatmap((16, 16, 16), [(2, 2, 2, 0.9), (14, 14, 14, 0.8)])
    peaks = detect.extract_peaks_single_frame(hm, threshold=0.3, min_distance_um=5.0, scale_down=SCALE_DOWN)
    assert len(peaks) == 2
    print("test_extract_peaks_far_apart_both_kept: PASS")


def test_extract_peaks_empty_heatmap():
    hm = np.zeros((8, 8, 8), dtype=np.float32)
    peaks = detect.extract_peaks_single_frame(hm, threshold=0.3, min_distance_um=5.0, scale_down=SCALE_DOWN)
    assert len(peaks) == 0
    print("test_extract_peaks_empty_heatmap: PASS")


def test_extract_peaks_rejects_wrong_ndim():
    hm = np.zeros((2, 8, 8, 8), dtype=np.float32)
    raised = False
    try:
        detect.extract_peaks_single_frame(hm, threshold=0.3, min_distance_um=5.0, scale_down=SCALE_DOWN)
    except ValueError:
        raised = True
    assert raised
    print("test_extract_peaks_rejects_wrong_ndim: PASS")


def test_peaks_to_native_coords_roundtrip():
    peaks_down = pd.DataFrame({"z": [4], "y": [10], "x": [4], "score": [0.9]})
    native = detect.peaks_to_native_coords(peaks_down, DOWNSAMPLE)
    # dz=1 -> z unchanged; y*4=40; x*4=16
    assert native.iloc[0]["z"] == 4
    assert native.iloc[0]["y"] == 40
    assert native.iloc[0]["x"] == 16
    assert np.isclose(native.iloc[0]["score"], 0.9)  # passes through unchanged
    print("test_peaks_to_native_coords_roundtrip: PASS")


def test_peaks_to_native_coords_empty():
    empty = pd.DataFrame({"z": [], "y": [], "x": [], "score": []})
    out = detect.peaks_to_native_coords(empty, DOWNSAMPLE)
    assert len(out) == 0
    print("test_peaks_to_native_coords_empty: PASS")


def test_detect_frame_end_to_end_on_fixture():
    # SYNTHETIC FIXTURE ONLY -- deliberately, per the fixture-vs-real-data
    # separation this test's own name commits to. threshold=0.0 is safe
    # here specifically because the fixture is tiny (guarded by the assert
    # below); do not point this test at real competition data with
    # threshold=0.0, that combination is what makes peak_local_max +
    # nms_3d expensive on a real-sized volume.
    samples = bio_io.list_samples(ROOT)
    s = samples[0]
    vol = bio_io.open_zarr_volume(s.zarr_path)
    assert np.prod(vol.shape[1:]) <= 64 * 64 * 64, (
        "this test uses threshold=0.0, which is only safe on a small fixture -- "
        f"got a frame of size {vol.shape[1:]}, which is too large for that"
    )

    raw_frame = np.asarray(vol[0])[None]  # (1, Z, Y, X) -- preprocess_window expects a leading axis
    frame_processed = prep.preprocess_window(raw_frame, DOWNSAMPLE)[0]  # (Zd, Yd, Xd)

    config = DetectorConfig(unet_out_channels=8, unet_layers=(8, 16), gradient_checkpointing=False)
    model = Detector3D(config)

    native_frame_shape = vol.shape[1:]  # (Z, Y, X)
    result = detect.detect_frame(
        model, frame_processed, DOWNSAMPLE, bio_io.NATIVE_SCALE,
        threshold=0.0,  # untrained model -> accept everything so we get *some* peaks to check bounds on
        min_distance_um=1.0,
    )
    assert set(result.columns) >= {"z", "y", "x", "score"}
    if len(result) > 0:
        assert (result["z"] >= 0).all() and (result["z"] < native_frame_shape[0]).all()
        assert (result["y"] >= 0).all() and (result["y"] < native_frame_shape[1]).all()
        assert (result["x"] >= 0).all() and (result["x"] < native_frame_shape[2]).all()
    print("test_detect_frame_end_to_end_on_fixture: PASS", "n_peaks=", len(result))


if __name__ == "__main__":
    test_extract_peaks_finds_isolated_peak()
    test_extract_peaks_below_threshold_dropped()
    test_extract_peaks_nms_suppresses_nearby_weaker_peak()
    test_extract_peaks_far_apart_both_kept()
    test_extract_peaks_empty_heatmap()
    test_extract_peaks_rejects_wrong_ndim()
    test_peaks_to_native_coords_roundtrip()
    test_peaks_to_native_coords_empty()
    test_detect_frame_end_to_end_on_fixture()
    print("\nALL PASS")
