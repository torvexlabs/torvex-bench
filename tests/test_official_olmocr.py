from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from torvex_bench.harnesses import official_olmocr as mod
from torvex_bench.harnesses.olmocr_eval import OlmOCRPredictionSummary


def test_parse_average_score() -> None:
    stdout = """
Final Summary with 95% Confidence Intervals:
torvex_extract       : Average Score: 42.3% ± 1.0% (average of per-JSONL scores)
"""
    assert mod.parse_average_score(stdout) == 0.423


def test_parse_average_score_returns_none_when_missing() -> None:
    assert mod.parse_average_score("no score here") is None


def test_parse_total_tests() -> None:
    stdout = """
Candidate: torvex_extract
  Average Score: 50.0% (95% CI: [40.0%, 60.0%]) over 25 tests.
"""
    assert mod.parse_total_tests(stdout) == 25


def test_parse_results_by_jsonl() -> None:
    stdout = """
    Results by JSONL file:
        old_scans.jsonl                : 50.0% (2/4 tests)
        table_tests.jsonl              : 25.0% (1/4 tests)
"""

    result = mod.parse_results_by_jsonl(stdout)

    assert result == {
        "old_scans.jsonl": {
            "pass_rate": 0.5,
            "passed": 2,
            "total": 4,
        },
        "table_tests.jsonl": {
            "pass_rate": 0.25,
            "passed": 1,
            "total": 4,
        },
    }


def test_resolve_olmocr_python_accepts_explicit_path(tmp_path: Path) -> None:
    explicit = tmp_path / "python.exe"

    assert mod.resolve_olmocr_python(explicit) == explicit


def test_clean_olmocr_work_dir_removes_generated_dir(tmp_path: Path) -> None:
    work_dir = tmp_path / "olmocr_work"
    work_dir.mkdir()
    (work_dir / "file.txt").write_text("x", encoding="utf-8")

    mod.clean_olmocr_work_dir(work_dir)

    assert not work_dir.exists()


def test_clean_olmocr_work_dir_rejects_unsafe_path() -> None:
    with pytest.raises(ValueError, match="unsafe"):
        mod.clean_olmocr_work_dir(Path("benchmarks"))


def test_run_official_olmocr_eval_builds_expected_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python_bin = tmp_path / "python.exe"
    python_bin.write_text("", encoding="utf-8")
    bench_data = tmp_path / "bench_data"
    bench_data.mkdir()

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = mod.run_official_olmocr_eval(
        bench_data_path=bench_data,
        python_bin=python_bin,
        candidate="torvex_extract",
    )

    assert result.returncode == 0
    command = calls[0][0]
    assert command[:3] == [str(python_bin), "-m", "olmocr.bench.benchmark"]
    assert "--dir" in command
    assert str(bench_data.resolve()) in command
    assert "--candidate" in command
    assert "torvex_extract" in command


def test_run_official_olmocr_orchestrates_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work_dir = tmp_path / "olmocr"
    data_dir = work_dir / "bench_data"
    prediction_dir = data_dir / "torvex_extract"

    calls = {
        "prepare": 0,
        "generate": 0,
        "eval": 0,
    }

    def fake_prepare_olmocr_bench(**kwargs):
        calls["prepare"] += 1
        assert kwargs["work_dir"] == work_dir
        assert kwargs["limit"] == 3
        assert kwargs["track"] == "non_math"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir / "sample_manifest.jsonl"

    def fake_generate_olmocr_predictions(**kwargs):
        calls["generate"] += 1
        assert kwargs["work_dir"] == work_dir
        assert kwargs["limit"] == 3
        assert kwargs["track"] == "non_math"
        assert kwargs["overwrite"] is True
        assert kwargs["device"] == "gpu"
        prediction_dir.mkdir(parents=True, exist_ok=True)

        return OlmOCRPredictionSummary(
            requested=3,
            processed=3,
            predictions_written=3,
            empty_predictions_written=0,
            skipped_existing=0,
            errors=0,
            prediction_dir=prediction_dir,
            normalized_dir=None,
            raw_dir=None,
        )

    def fake_run_official_olmocr_eval(**kwargs):
        calls["eval"] += 1
        assert kwargs["bench_data_path"] == data_dir
        return subprocess.CompletedProcess(
            args=["python"],
            returncode=0,
            stdout=(
                "Candidate: torvex_extract\n"
                "  Average Score: 50.0% (95% CI: [40.0%, 60.0%]) over 3 tests.\n"
                "\n"
                "Final Summary with 95% Confidence Intervals:\n"
                "torvex_extract       : Average Score: 50.0% ± 1.0% "
                "(average of per-JSONL scores)\n"
                "\n"
                "    Results by JSONL file:\n"
                "        old_scans.jsonl                : 50.0% (1/2 tests)\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(mod, "prepare_olmocr_bench", fake_prepare_olmocr_bench)
    monkeypatch.setattr(mod, "generate_olmocr_predictions", fake_generate_olmocr_predictions)
    monkeypatch.setattr(mod, "run_official_olmocr_eval", fake_run_official_olmocr_eval)

    summary = mod.run_official_olmocr(
        work_dir=work_dir,
        limit=3,
        track="non_math",
        clean=True,
        save_normalized=False,
        device="gpu",
    )

    assert calls == {"prepare": 1, "generate": 1, "eval": 1}
    assert summary.eval_returncode == 0
    assert summary.average_score == 0.5
    assert summary.total_tests == 3
    assert summary.results_by_jsonl["old_scans.jsonl"]["passed"] == 1
    assert summary.prediction_summary["predictions_written"] == 3
    assert summary.summary_path.exists()

    payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "olmOCR-Bench"
    assert payload["engine"] == "torvex_extract"
    assert payload["track"] == "non_math"
    assert payload["average_score"] == 0.5


def test_run_official_olmocr_rejects_invalid_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="limit"):
        mod.run_official_olmocr(
            work_dir=tmp_path / "olmocr",
            limit=0,
        )


def test_run_official_olmocr_rejects_invalid_device(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="device"):
        mod.run_official_olmocr(
            work_dir=tmp_path / "olmocr",
            limit=1,
            device="tpu",
        )