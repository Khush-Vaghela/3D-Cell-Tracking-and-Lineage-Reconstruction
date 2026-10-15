import os
import sys

import numpy as np
import pandas as pd

import node_validation as nv
import data_io as bio_io

SCALE = (1.625, 0.40625, 0.40625)
ROOT = "/kaggle/input/competitions/biohub-cell-tracking-during-development/train"

if not os.path.isdir(ROOT):
    import make_fixtures
    make_fixtures.main()


def test_perfect_match_recall_and_precision_are_1():
    gt = pd.DataFrame({"t": [0, 0], "z": [4, 10], "y": [8, 20], "x": [8, 20]})
    pred = gt.copy()
    result = nv.compute_node_recall(pred, gt, SCALE, max_distance=1.0)
    assert result.recall == 1.0
    assert result.precision == 1.0
    assert result.n_matched == 2
    print("test_perfect_match_recall_and_precision_are_1: PASS")


def test_partial_recall():
    gt = pd.DataFrame({"t": [0, 0, 0], "z": [4, 10, 20], "y": [8, 20, 30], "x": [8, 20, 30]})
    # only predict 2 of the 3 GT cells, exactly on target
    pred = pd.DataFrame({"t": [0, 0], "z": [4, 10], "y": [8, 20], "x": [8, 20]})
    result = nv.compute_node_recall(pred, gt, SCALE, max_distance=1.0)
    assert np.isclose(result.recall, 2 / 3)
    assert result.precision == 1.0  # both predictions were correct
    print("test_partial_recall: PASS", result)


def test_extra_false_positive_hurts_precision_not_recall():
    gt = pd.DataFrame({"t": [0], "z": [4], "y": [8], "x": [8]})
    pred = pd.DataFrame({"t": [0, 0], "z": [4, 50], "y": [8, 50], "x": [8, 50]})  # one correct, one FP far away
    result = nv.compute_node_recall(pred, gt, SCALE, max_distance=1.0)
    assert result.recall == 1.0    # the one GT node was found
    assert np.isclose(result.precision, 0.5)  # only 1 of 2 predictions was correct
    print("test_extra_false_positive_hurts_precision_not_recall: PASS", result)


def test_no_gt_nodes_returns_nan():
    gt = pd.DataFrame({"t": [], "z": [], "y": [], "x": []})
    pred = pd.DataFrame({"t": [0], "z": [4], "y": [8], "x": [8]})
    result = nv.compute_node_recall(pred, gt, SCALE)
    assert result.recall != result.recall  # NaN
    print("test_no_gt_nodes_returns_nan: PASS")


def test_no_predictions_returns_zero_recall():
    gt = pd.DataFrame({"t": [0], "z": [4], "y": [8], "x": [8]})
    pred = pd.DataFrame({"t": [], "z": [], "y": [], "x": []})
    result = nv.compute_node_recall(pred, gt, SCALE)
    assert result.recall == 0.0
    assert result.precision != result.precision  # NaN -- no predictions to have a precision over
    print("test_no_predictions_returns_zero_recall: PASS")


def test_far_apart_nodes_dont_match():
    gt = pd.DataFrame({"t": [0], "z": [4], "y": [8], "x": [8]})
    pred = pd.DataFrame({"t": [0], "z": [4], "y": [8], "x": [200]})  # far away in x
    result = nv.compute_node_recall(pred, gt, SCALE, max_distance=7.0)
    assert result.recall == 0.0
    assert result.precision == 0.0
    print("test_far_apart_nodes_dont_match: PASS")


def test_integration_with_real_fixture_geff():
    samples = bio_io.list_samples(ROOT)
    s = samples[0]
    geff = bio_io.load_geff(s.geff_path)
    gt_nodes = geff.nodes[["t", "z", "y", "x"]]

    # "predict" by copying GT exactly -- should give perfect recall, confirming
    # the real .geff-derived data plugs into this module correctly.
    result = nv.compute_node_recall(gt_nodes, gt_nodes, bio_io.NATIVE_SCALE, max_distance=1.0)
    assert result.recall == 1.0
    assert result.precision == 1.0
    print("test_integration_with_real_fixture_geff: PASS", result)


if __name__ == "__main__":
    test_perfect_match_recall_and_precision_are_1()
    test_partial_recall()
    test_extra_false_positive_hurts_precision_not_recall()
    test_no_gt_nodes_returns_nan()
    test_no_predictions_returns_zero_recall()
    test_far_apart_nodes_dont_match()
    test_integration_with_real_fixture_geff()
    print("\nALL PASS")
