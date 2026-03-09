"""Tests for results schema roundtrip serialization."""

import json

import pytest

from bench.results_schema import (
    BenchmarkResult,
    EnvironmentSnapshot,
    RunConfig,
    SummaryRow,
    TrialRecord,
)


class TestTrialRecord:
    def test_default_values(self):
        t = TrialRecord()
        assert t.trial_idx == 0
        assert t.per_token_ms == []
        assert t.ttft_ms == 0.0

    def test_with_data(self):
        t = TrialRecord(
            trial_idx=3,
            ttft_ms=15.5,
            end_to_end_ms=500.0,
            throughput_tok_s=25.6,
            per_token_ms=[3.0, 3.1, 2.9],
            generated_tokens=3,
        )
        assert t.trial_idx == 3
        assert len(t.per_token_ms) == 3


class TestBenchmarkResult:
    def test_roundtrip(self):
        env = EnvironmentSnapshot(os="Linux", python_version="3.11.0", cpu_cores=8)
        config = RunConfig(
            run_id="test_001",
            backend="hf",
            device="cpu",
            model_id="tiny-gpt2",
            prompt_length=64,
            output_length=32,
        )
        trials = [
            TrialRecord(trial_idx=0, ttft_ms=10.0, per_token_ms=[3.0, 3.1]),
            TrialRecord(trial_idx=1, ttft_ms=11.0, per_token_ms=[3.2, 2.9]),
        ]
        result = BenchmarkResult(config=config, env=env, trials=trials)

        # Serialize
        json_str = result.to_json()
        data = json.loads(json_str)

        # Deserialize
        restored = BenchmarkResult.from_dict(data)

        assert restored.config.run_id == "test_001"
        assert restored.config.backend == "hf"
        assert restored.env.os == "Linux"
        assert len(restored.trials) == 2
        assert restored.trials[0].ttft_ms == 10.0
        assert restored.trials[1].per_token_ms == [3.2, 2.9]

    def test_to_dict_keys(self):
        result = BenchmarkResult()
        d = result.to_dict()
        assert "config" in d
        assert "env" in d
        assert "trials" in d
        assert "metadata" in d

    def test_empty_result(self):
        result = BenchmarkResult()
        json_str = result.to_json()
        restored = BenchmarkResult.from_dict(json.loads(json_str))
        assert restored.trials == []


class TestSummaryRow:
    def test_header_and_values(self):
        row = SummaryRow(run_id="test", ttft_mean_ms=10.5, per_token_mean_ms=3.2)
        header = row.header()
        values = row.values()
        assert "run_id" in header
        assert "ttft_mean_ms" in header
        assert len(header) == len(values)
        assert values[header.index("run_id")] == "test"
        assert values[header.index("ttft_mean_ms")] == 10.5
