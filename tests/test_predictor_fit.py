"""Tests for the latency predictor fit module."""

import numpy as np
import pytest

from analysis.predictor_fit import build_features, fit_predictor


class TestBuildFeatures:
    def test_basic_shape(self):
        rows = [
            {"prompt_length": 128, "per_token_mean_ms": 5.0},
            {"prompt_length": 256, "per_token_mean_ms": 6.0},
            {"prompt_length": 512, "per_token_mean_ms": 8.0},
        ]
        X, y = build_features(rows, bandwidth_gb_s=50.0, model_weight_bytes=14e9)
        assert X.shape == (3, 3)
        assert y.shape == (3,)

    def test_skips_invalid_rows(self):
        rows = [
            {"prompt_length": 0, "per_token_mean_ms": 5.0},
            {"prompt_length": 128, "per_token_mean_ms": 0},
            {"prompt_length": 256, "per_token_mean_ms": 6.0},
        ]
        X, y = build_features(rows, bandwidth_gb_s=50.0, model_weight_bytes=14e9)
        assert X.shape[0] == 1  # Only the valid row
        assert y[0] == 6.0

    def test_constant_column(self):
        rows = [
            {"prompt_length": 128, "per_token_mean_ms": 5.0},
            {"prompt_length": 256, "per_token_mean_ms": 6.0},
        ]
        X, y = build_features(rows, bandwidth_gb_s=50.0, model_weight_bytes=14e9)
        # Third column should be all 1.0 (constant/overhead term)
        np.testing.assert_array_equal(X[:, 2], [1.0, 1.0])


class TestFitPredictor:
    def test_perfect_fit(self):
        X = np.array([
            [1.0, 0.5, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 4.0, 1.0],
        ])
        # y = 2*x0 + 3*x1 + 1*x2
        y = 2 * X[:, 0] + 3 * X[:, 1] + 1 * X[:, 2]
        result = fit_predictor(X, y)
        assert result["r2"] > 0.99
        assert result["mae"] < 0.01

    def test_insufficient_data(self):
        X = np.array([[1.0, 0.5, 1.0]])
        y = np.array([5.0])
        result = fit_predictor(X, y)
        assert result["n_samples"] == 1

    def test_output_keys(self):
        X = np.array([
            [1.0, 0.5, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
        ])
        y = np.array([3.0, 4.0, 6.0])
        result = fit_predictor(X, y)
        assert "coefficients" in result
        assert "mae" in result
        assert "mape" in result
        assert "r2" in result
        assert "n_samples" in result
        assert len(result["coefficients"]) == 3

    def test_nonnegative_coefficients(self):
        np.random.seed(42)
        X = np.random.rand(20, 3)
        X[:, 2] = 1.0
        y = 2 * X[:, 0] + 3 * X[:, 1] + 0.5 + np.random.randn(20) * 0.01
        result = fit_predictor(X, y)
        assert all(c >= 0 for c in result["coefficients"])
