"""Unit tests for boundary F1 evaluation metrics."""

from __future__ import annotations

import pytest

from vlm_video.evaluation.boundary_f1 import boundary_f1, match_boundaries


class TestMatchBoundaries:
    def test_perfect_match(self):
        tp, fp, fn = match_boundaries([10.0, 60.0], [10.0, 60.0], tolerance=5)
        assert tp == 2
        assert fp == 0
        assert fn == 0

    def test_within_tolerance(self):
        # Prediction is 3 s off, tolerance is 5 s → should match
        tp, fp, fn = match_boundaries([13.0], [10.0], tolerance=5)
        assert tp == 1
        assert fp == 0
        assert fn == 0

    def test_outside_tolerance(self):
        # Prediction is 6 s off, tolerance is 5 s → no match
        tp, fp, fn = match_boundaries([16.0], [10.0], tolerance=5)
        assert tp == 0
        assert fp == 1
        assert fn == 1

    def test_false_positive(self):
        # Extra prediction with no matching GT
        tp, fp, fn = match_boundaries([10.0, 100.0], [10.0], tolerance=5)
        assert tp == 1
        assert fp == 1
        assert fn == 0

    def test_false_negative(self):
        # Missing prediction
        tp, fp, fn = match_boundaries([10.0], [10.0, 60.0], tolerance=5)
        assert tp == 1
        assert fp == 0
        assert fn == 1

    def test_empty_predictions(self):
        tp, fp, fn = match_boundaries([], [10.0, 60.0], tolerance=5)
        assert tp == 0
        assert fp == 0
        assert fn == 2

    def test_empty_ground_truth(self):
        tp, fp, fn = match_boundaries([10.0, 60.0], [], tolerance=5)
        assert tp == 0
        assert fp == 2
        assert fn == 0

    def test_both_empty(self):
        tp, fp, fn = match_boundaries([], [], tolerance=5)
        assert tp == 0
        assert fp == 0
        assert fn == 0

    def test_each_gt_matched_at_most_once(self):
        # Two predictions both close to the same GT boundary
        tp, fp, fn = match_boundaries([10.0, 11.0], [10.0], tolerance=5)
        assert tp == 1
        assert fp == 1
        assert fn == 0


class TestBoundaryF1:
    def test_perfect_f1(self):
        result = boundary_f1([10.0, 60.0], [10.0, 60.0], tolerances=[5])
        assert result[5]["F1"] == pytest.approx(1.0)
        assert result[5]["P"] == pytest.approx(1.0)
        assert result[5]["R"] == pytest.approx(1.0)

    def test_zero_f1_no_match(self):
        result = boundary_f1([100.0], [10.0], tolerances=[5])
        assert result[5]["F1"] == pytest.approx(0.0)

    def test_multiple_tolerances(self):
        result = boundary_f1([10.0], [15.0], tolerances=[5, 10])
        # 5 s tolerance: |10 - 15| = 5, exactly on boundary
        assert result[5]["F1"] == pytest.approx(1.0)
        assert result[10]["F1"] == pytest.approx(1.0)

    def test_recall_higher_than_precision(self):
        # Predictions miss one GT but add no false positives
        result = boundary_f1([10.0], [10.0, 60.0], tolerances=[5])
        assert result[5]["R"] == pytest.approx(0.5)
        assert result[5]["P"] == pytest.approx(1.0)
        f1 = result[5]["F1"]
        assert 0.0 < f1 < 1.0

    def test_default_tolerances(self):
        result = boundary_f1([10.0], [10.0])
        assert 5.0 in result
        assert 10.0 in result

    def test_empty_preds_and_gt(self):
        result = boundary_f1([], [], tolerances=[5])
        assert result[5]["P"] == pytest.approx(0.0)
        assert result[5]["R"] == pytest.approx(0.0)
        assert result[5]["F1"] == pytest.approx(0.0)

    def test_only_false_positives(self):
        result = boundary_f1([10.0, 20.0, 30.0], [], tolerances=[5])
        assert result[5]["P"] == pytest.approx(0.0)
        assert result[5]["R"] == pytest.approx(0.0)

    def test_returns_rounded_values(self):
        result = boundary_f1([10.0], [10.0, 60.0], tolerances=[5])
        # Values should be rounded to 4 decimal places
        for key in ("P", "R", "F1"):
            val = result[5][key]
            assert val == round(val, 4)
