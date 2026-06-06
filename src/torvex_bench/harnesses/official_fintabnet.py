from __future__ import annotations

"""
Official FinTabNet benchmark runner.

This module owns the full official docling-eval flow for FinTabNet.

It is intentionally separate from cli.py because cli.py should stay thin:
    parse args -> call production function -> print result

High-level flow:
    1. Create a clean work directory.
    2. Ask docling-eval to create a matching GT dataset for the requested limit.
    3. Generate Torvex predictions in docling-eval File-provider format.
    4. Ask docling-eval to create the evaluation dataset.
    5. Ask docling-eval to run official table_structure evaluation.
    6. Read the official result JSON and return a compact summary.

Important:
    This module does not implement TEDS.
    docling-eval remains the official scorer.
"""

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torvex_bench.harnesses.docling_eval_fintabnet import (
    FinTabNetPredictionSummary,
    generate_fintabnet_predictions,
)


DEFAULT_WORK_DIR = Path("benchmarks/docling_eval/FinTabNet_dev")
DEFAULT_SUMMARY_FILENAME = "official_fintabnet_summary.json"


@dataclass(frozen=True)
class OfficialFinTabNetSummary:
    """
    Compact summary of one official FinTabNet run.

    This is separate from the full docling-eval result JSON.
    The full official result remains in:
        <work_dir>/results/evaluations/table_structure/
            evaluation_FinTabNet_table_structure.json
    """

    limit: int
    work_dir: Path
    prediction_summary: FinTabNetPredictionSummary
    evaluated_samples: int
    teds_mean: float
    teds_struct_mean: float
    rejected_samples: dict[str, int]
    official_result_path: Path
    summary_path: Path


def _run_command(command: list[str]) -> None:
    """
    Run one external command and fail immediately if it fails.

    We use the official docling-eval CLI instead of importing evaluator internals.
    That keeps the benchmark behavior close to how reviewers would run it.
    """
    print()
    print("[torvex-bench] running:")
    print("  " + " ".join(command))

    subprocess.run(command, check=True)


def _safe_clean_work_dir(work_dir: Path) -> None:
    """
    Remove generated work directory safely.

    Guard:
        Do not allow accidental deletion of the permanent FinTabNet GT folder.
        The default work dir is FinTabNet_dev, which is safe to delete.
    """
    permanent_dir = Path("benchmarks/docling_eval/FinTabNet").resolve()
    resolved_work_dir = work_dir.resolve()

    if resolved_work_dir == permanent_dir:
        raise ValueError(
            "Refusing to clean benchmarks/docling_eval/FinTabNet directly. "
            "Use a dev work directory such as benchmarks/docling_eval/FinTabNet_dev."
        )

    shutil.rmtree(work_dir, ignore_errors=True)


def _official_result_path(work_dir: Path) -> Path:
    """
    Return the official docling-eval result JSON path for FinTabNet table_structure.
    """
    return (
        work_dir
        / "results"
        / "evaluations"
        / "table_structure"
        / "evaluation_FinTabNet_table_structure.json"
    )


def _load_official_result(result_path: Path) -> dict[str, Any]:
    """
    Read docling-eval's official evaluation JSON.
    """
    if not result_path.exists():
        raise FileNotFoundError(
            f"Official result JSON not found: {result_path}. "
            "docling-eval evaluate probably did not finish successfully."
        )

    return json.loads(result_path.read_text(encoding="utf-8"))


def _write_compact_summary(
    *,
    summary: OfficialFinTabNetSummary,
) -> None:
    """
    Save a small human-readable summary next to the official result.

    This does not replace the official docling-eval JSON.
    It only makes quick inspection easier.
    """
    payload = {
        "benchmark": "FinTabNet",
        "engine": "torvex_extract",
        "limit": summary.limit,
        "work_dir": str(summary.work_dir),
        "evaluated_samples": summary.evaluated_samples,
        "teds_mean": summary.teds_mean,
        "teds_struct_mean": summary.teds_struct_mean,
        "rejected_samples": summary.rejected_samples,
        "prediction_summary": {
            "requested": summary.prediction_summary.requested,
            "processed": summary.prediction_summary.processed,
            "predictions_written": summary.prediction_summary.predictions_written,
            "missing_tables": summary.prediction_summary.missing_tables,
            "skipped_existing": summary.prediction_summary.skipped_existing,
            "errors": summary.prediction_summary.errors,
            "prediction_dir": str(summary.prediction_summary.prediction_dir),
            "normalized_dir": (
                str(summary.prediction_summary.normalized_dir)
                if summary.prediction_summary.normalized_dir is not None
                else None
            ),
        },
        "official_result_path": str(summary.official_result_path),
    }

    summary.summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_official_fintabnet(
    *,
    limit: int,
    work_dir: Path = DEFAULT_WORK_DIR,
    clean: bool = True,
    save_normalized: bool = False,
) -> OfficialFinTabNetSummary:
    """
    Run the full official FinTabNet table_structure benchmark for Torvex.

    Args:
        limit:
            Number of FinTabNet test samples to evaluate.
            Example: 25, 100, 1000.

        work_dir:
            Isolated generated-artifact folder.
            Default:
                benchmarks/docling_eval/FinTabNet_dev

        clean:
            If True, delete work_dir before running.
            This prevents stale GT/eval/results/predictions from mixing across runs.

        save_normalized:
            Debug option.
            If True, save normalized Torvex JSON under results/raw.
            If False, only prediction JSONs and official docling-eval artifacts are saved.

    Returns:
        OfficialFinTabNetSummary with the clean metrics and paths.
    """
    if limit <= 0:
        raise ValueError("limit must be a positive integer")

    if clean:
        _safe_clean_work_dir(work_dir)

    work_dir.mkdir(parents=True, exist_ok=True)

    gt_dir = work_dir / "gt_dataset"
    gt_test_dir = gt_dir / "test"
    prediction_dir = work_dir / "predictions" / "torvex_extract"
    eval_dataset_dir = work_dir / "eval_dataset"
    results_dir = work_dir / "results"

    # 1. Create matching official GT for this exact limit.
    _run_command(
        [
            "docling-eval",
            "create-gt",
            "--benchmark",
            "FinTabNet",
            "--output-dir",
            str(work_dir),
            "--split",
            "test",
            "--end-index",
            str(limit),
            "--no-do-visualization",
        ]
    )

    # 2. Generate Torvex prediction JSON files from the GT parquet rows.
    prediction_summary = generate_fintabnet_predictions(
        gt_dir=gt_test_dir,
        prediction_dir=prediction_dir,
        limit=None,
        overwrite=True,
        save_normalized=save_normalized,
    )

    # 3. Build docling-eval evaluation dataset using File provider.
    _run_command(
        [
            "docling-eval",
            "create-eval",
            "--benchmark",
            "FinTabNet",
            "--gt-dir",
            str(gt_dir),
            "--output-dir",
            str(work_dir),
            "--prediction-provider",
            "File",
            "--file-prediction-format",
            "json",
            "--file-source-path",
            str(prediction_dir),
            "--file-use-ground-truth-images",
            "--split",
            "test",
            "--no-do-visualization",
        ]
    )

    # 4. Run official table_structure evaluation.
    _run_command(
        [
            "docling-eval",
            "evaluate",
            "--benchmark",
            "FinTabNet",
            "--modality",
            "table_structure",
            "--input-dir",
            str(eval_dataset_dir),
            "--output-dir",
            str(results_dir),
            "--split",
            "test",
        ]
    )

    # 5. Read official result JSON and make compact summary.
    official_result_path = _official_result_path(work_dir)
    official_result = _load_official_result(official_result_path)

    summary_path = results_dir / DEFAULT_SUMMARY_FILENAME

    summary = OfficialFinTabNetSummary(
        limit=limit,
        work_dir=work_dir,
        prediction_summary=prediction_summary,
        evaluated_samples=int(official_result["evaluated_samples"]),
        teds_mean=float(official_result["TEDS"]["mean"]),
        teds_struct_mean=float(official_result["TEDS_struct"]["mean"]),
        rejected_samples=dict(official_result["rejected_samples"]),
        official_result_path=official_result_path,
        summary_path=summary_path,
    )

    _write_compact_summary(summary=summary)

    return summary