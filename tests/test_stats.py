"""Tests for statistical utilities."""

import numpy as np
import pytest

from bench.utils.stats import bootstrap_ci, iqr_filter, robust_summary, steady_state_latencies


class TestIQRFilter:
    def test_no_outliers(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = iqr_filter(data)
        np.testing.assert_array_equal(result, data)

    def test_removes_outliers(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        result = iqr_filter(data)
        assert 100.0 not in result
        assert len(result) < len(data)

    def test_small_array_unchanged(self):
        data = np.array([1.0, 2.0])
        result = iqr_filter(data)
        np.testing.assert_array_equal(result, data)

    def test_empty_array(self):
        data = np.array([])
        result = iqr_filter(data)
        assert len(result) == 0

    def test_all_same_values(self):
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = iqr_filter(data)
        np.testing.assert_array_equal(result, data)


class TestBootstrapCI:
    def test_basic_ci(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lo, hi = bootstrap_ci(data)
        assert lo < hi
        assert lo <= np.mean(data)
        assert hi >= np.mean(data)

    def test_single_value(self):
        data = np.array([42.0])
        lo, hi = bootstrap_ci(data)
        assert lo == 42.0
        assert hi == 42.0

    def test_deterministic(self):
        data = np.arange(100, dtype=np.float64)
        lo1, hi1 = bootstrap_ci(data, rng_seed=123)
        lo2, hi2 = bootstrap_ci(data, rng_seed=123)
        assert lo1 == lo2
        assert hi1 == hi2

    def test_narrow_ci_for_tight_data(self):
        data = np.array([10.0, 10.01, 9.99, 10.02, 9.98])
        lo, hi = bootstrap_ci(data)
        assert hi - lo < 0.1


class TestRobustSummary:
    def test_basic_summary(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = robust_summary(values)
        assert result["mean"] == pytest.approx(3.0, abs=0.5)
        assert result["median"] == pytest.approx(3.0, abs=0.5)
        assert result["n"] >= 4
        assert "ci_low" in result
        assert "ci_high" in result

    def test_empty_values(self):
        result = robust_summary([])
        assert result["mean"] == 0.0
        assert result["n"] == 0

    def test_without_iqr(self):
        values = [1.0, 2.0, 3.0, 100.0]
        with_iqr = robust_summary(values, apply_iqr=True)
        without_iqr = robust_summary(values, apply_iqr=False)
        assert without_iqr["mean"] > with_iqr["mean"]


class TestSteadyState:
    def test_skip_first_two(self):
        traces = [
            [10.0, 8.0, 3.0, 3.1, 2.9],
            [9.0, 7.5, 3.2, 3.0, 3.1],
        ]
        result = steady_state_latencies(traces, skip=2)
        assert len(result) == 6  # 3 + 3 after skipping 2 from each
        assert all(v < 5.0 for v in result)

    def test_skip_zero(self):
        traces = [[1.0, 2.0, 3.0]]
        result = steady_state_latencies(traces, skip=0)
        assert len(result) == 3

    def test_empty_traces(self):
        result = steady_state_latencies([], skip=2)
        assert result == []

    def test_skip_more_than_length(self):
        traces = [[1.0, 2.0]]
        result = steady_state_latencies(traces, skip=5)
        assert result == []
