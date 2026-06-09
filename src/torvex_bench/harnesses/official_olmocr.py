"""
Official olmOCR-Bench evaluator wrapper.

Purpose
-------
Run the official olmOCR-Bench evaluator on Torvex predictions.

Flow
----
1. Prepare official bench_data folder:
      *.jsonl
      pdfs/
      sample_manifest.jsonl

2. Generate Torvex Markdown predictions:
      bench_data/torvex_extract/<pdf_stem>_pg<page>_repeat1.md

3. Call official evaluator:
      python -m olmocr.bench.benchmark
        --dir <bench_data>
        --candidate torvex_extract

4. Save stdout/stderr and compact summary.json.

No metric is implemented here.
The official olmOCR evaluator computes pass rates.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from torvex_bench.datasets.olmocr import (
    TRACK_NON_MATH,
    bench_data_dir,
    prepare_olmocr_bench,
)
from torvex_bench.harnesses.olmocr_eval import (
    DEFAULT_ENGINE_NAME,
    OlmOCRPredictionSummary,
    generate_olmocr_predictions,
)


DEFAULT_WORK_DIR = Path("benchmarks/olmocr/olmOCR_Bench_non_math")
DEFAULT_SUMMARY_FILENAME = "summary.json"


@dataclass(slots=True)
class OfficialOlmOCRSummary:
    benchmark: str
    engine: str
    track: str
    limit: int
    device: str
    work_dir: Path
    bench_data_dir: Path
    predictions_dir: Path
    summary_path: Path
    stdout_path: Path
    stderr_path: Path
    eval_returncode: int
    prediction_summary: dict[str, Any]
    average_score: float | None
    total_tests: int | None
    results_by_jsonl: dict[str, dict[str, Any]]
    stdout_tail: str


def resolve_olmocr_python(python_bin: str | Path | None = None) -> Path:
    """
    Resolve isolated olmOCR Python executable.

    Priority:
        1. explicit python_bin
        2. OLMOCR_PYTHON env var
        3. local Windows venv path
        4. local POSIX venv path
    """
    if python_bin:
        return Path(python_bin)

    env_value = os.getenv("OLMOCR_PYTHON")
    if env_value:
        return Path(env_value)

    windows_path = Path("data/venvs/olmocr/Scripts/python.exe")
    if windows_path.exists():
        return windows_path

    return Path("data/venvs/olmocr/bin/python")


def clean_olmocr_work_dir(work_dir: str | Path) -> None:
    """
    Clean generated olmOCR run artifacts while keeping nothing permanent.

    This work_dir is generated data only:
        benchmarks/olmocr/olmOCR_Bench_non_math
    """
    work_dir = Path(work_dir)

    guarded = [
        Path(".").resolve(),
        Path("benchmarks").resolve(),
        Path("benchmarks/olmocr").resolve(),
        Path("data").resolve(),
    ]

    resolved = work_dir.resolve()

    if resolved in guarded:
        raise ValueError(f"Refusing to clean unsafe work_dir: {work_dir}")

    if work_dir.exists():
        shutil.rmtree(work_dir)


def run_official_olmocr_eval(
    *,
    bench_data_path: str | Path,
    python_bin: str | Path | None = None,
    candidate: str = DEFAULT_ENGINE_NAME,
) -> subprocess.CompletedProcess[str]:
    """
    Call official olmOCR-Bench evaluator.
    """
    resolved_python = resolve_olmocr_python(python_bin)

    if not resolved_python.exists():
        raise FileNotFoundError(
            f"olmOCR Python executable not found: {resolved_python}. "
            "Create isolated env first: uv venv data/venvs/olmocr --python 3.11"
        )

    command = [
        str(resolved_python),
        "-m",
        "olmocr.bench.benchmark",
        "--dir",
        str(Path(bench_data_path).resolve()),
        "--candidate",
        candidate,
    ]

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")

    return subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        env=env,
    )


def parse_average_score(stdout: str) -> float | None:
    """
    Parse official final summary average score from stdout.

    Example line:
        torvex_extract : Average Score: 42.3% ± 1.0% ...
    """
    pattern = re.compile(rf"{re.escape(DEFAULT_ENGINE_NAME)}\s+:\s+Average Score:\s+([0-9.]+)%")
    match = pattern.search(stdout or "")

    if not match:
        return None

    return float(match.group(1)) / 100.0


def parse_total_tests(stdout: str) -> int | None:
    """
    Parse total test count from official stdout.

    Example line:
        Average Score: 42.3% (...) over 25 tests.
    """
    matches = re.findall(r"over\s+(\d+)\s+tests", stdout or "")

    if not matches:
        return None

    return int(matches[-1])


def parse_results_by_jsonl(stdout: str) -> dict[str, dict[str, Any]]:
    """
    Parse official per-JSONL result lines.

    Example:
        old_scans.jsonl                : 12.5% (1/8 tests)
    """
    results: dict[str, dict[str, Any]] = {}

    pattern = re.compile(
        r"^\s*([A-Za-z0-9_.-]+\.jsonl)\s*:\s*([0-9.]+)%\s*\((\d+)/(\d+)\s+tests\)",
        re.MULTILINE,
    )

    for match in pattern.finditer(stdout or ""):
        jsonl_name = match.group(1)
        results[jsonl_name] = {
            "pass_rate": float(match.group(2)) / 100.0,
            "passed": int(match.group(3)),
            "total": int(match.group(4)),
        }

    return results


def _tail_text(text: str, *, max_chars: int = 4000) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _write_summary(summary: OfficialOlmOCRSummary) -> None:
    summary.summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.summary_path.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def run_official_olmocr(
    *,
    work_dir: Path = DEFAULT_WORK_DIR,
    limit: int = 3,
    track: str = TRACK_NON_MATH,
    clean: bool = True,
    save_normalized: bool = False,
    device: str = "cpu",
    python_bin: str | Path | None = None,
    enable_formula: bool | None = None,
) -> OfficialOlmOCRSummary:
    """
    Generate Torvex predictions and run official olmOCR-Bench evaluation.
    """
    if limit <= 0:
        raise ValueError("limit must be a positive integer")

    if device not in {"cpu", "gpu"}:
        raise ValueError("device must be 'cpu' or 'gpu'")

    work_dir = Path(work_dir)

    if clean:
        clean_olmocr_work_dir(work_dir)

    # Prepare first so official JSONLs/PDFs exist.
    prepare_olmocr_bench(
        work_dir=work_dir,
        limit=limit,
        track=track,
        download_pdfs=True,
    )

    prediction_summary: OlmOCRPredictionSummary = generate_olmocr_predictions(
        work_dir=work_dir,
        limit=limit,
        track=track,
        overwrite=clean,
        save_raw=False,
        save_normalized=save_normalized,
        device=device,
        enable_formula=enable_formula,
    )

    data_dir = bench_data_dir(work_dir)
    predictions_dir = data_dir / DEFAULT_ENGINE_NAME
    summary_path = work_dir / DEFAULT_SUMMARY_FILENAME
    stdout_path = work_dir / "olmocr_eval_stdout.log"
    stderr_path = work_dir / "olmocr_eval_stderr.log"

    completed = run_official_olmocr_eval(
        bench_data_path=data_dir,
        python_bin=python_bin,
        candidate=DEFAULT_ENGINE_NAME,
    )

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")

    summary = OfficialOlmOCRSummary(
        benchmark="olmOCR-Bench",
        engine=DEFAULT_ENGINE_NAME,
        track=track,
        limit=limit,
        device=device,
        work_dir=work_dir,
        bench_data_dir=data_dir,
        predictions_dir=predictions_dir,
        summary_path=summary_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        eval_returncode=int(completed.returncode),
        prediction_summary=asdict(prediction_summary),
        average_score=parse_average_score(completed.stdout or ""),
        total_tests=parse_total_tests(completed.stdout or ""),
        results_by_jsonl=parse_results_by_jsonl(completed.stdout or ""),
        stdout_tail=_tail_text(completed.stdout or ""),
    )

    _write_summary(summary)

    if completed.returncode != 0:
        raise RuntimeError(
            f"official olmOCR evaluator failed with returncode={completed.returncode}. "
            f"See {stdout_path} and {stderr_path}."
        )

    return summary