from __future__ import annotations

"""
Official DocLayNetV1 benchmark runner.

This module owns the full official docling-eval flow for DocLayNetV1.

It is intentionally separate from cli.py because cli.py should stay thin:
    parse args -> call production function -> print result

High-level flow:
    1. Create a clean work directory.
    2. Ask docling-eval to create a matching DocLayNetV1 GT dataset.
    3. Generate Torvex predictions in docling-eval File-provider format.
    4. Ask docling-eval to create the evaluation dataset.
    5. Ask docling-eval to run official layout evaluation.
    6. Read the official result JSON and return a compact summary.

Important:
    This module does not implement mAP.
    docling-eval remains the official scorer.
"""

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torvex_bench.harnesses.docling_eval_doclaynet import (
    DocLayNetPredictionSummary,
    generate_doclaynet_predictions,
)


DEFAULT_WORK_DIR = Path("benchmarks/docling_eval/DocLayNetV1_dev")
DEFAULT_SUMMARY_FILENAME = "official_doclaynet_summary.json"


@dataclass(frozen=True)
class OfficialDocLayNetSummary:
    """
    Compact summary of one official DocLayNetV1 run.

    This is separate from the full docling-eval result JSON.

    The full official result remains in:
        <work_dir>/results/evaluations/layout/
            evaluation_DocLayNetV1_layout.json
    """

    limit: int
    device: str
    work_dir: Path
    prediction_summary: DocLayNetPredictionSummary
    evaluated_samples: int
    map_score: float
    map_50_mean: float
    map_75_mean: float
    map_mean: float
    weighted_map_50_mean: float
    rejected_samples: dict[str, int]
    true_labels: dict[str, int]
    pred_labels: dict[str, int]
    intersecting_labels: list[str]
    evaluations_per_class: list[dict[str, Any]]
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
        Do not allow accidental deletion of the permanent DocLayNetV1 folder.
        The default work dir is DocLayNetV1_dev, which is safe to delete.
    """
    permanent_dir = Path("benchmarks/docling_eval/DocLayNetV1").resolve()
    resolved_work_dir = work_dir.resolve()

    if resolved_work_dir == permanent_dir:
        raise ValueError(
            "Refusing to clean benchmarks/docling_eval/DocLayNetV1 directly. "
            "Use a dev work directory such as benchmarks/docling_eval/DocLayNetV1_dev."
        )

    if resolved_work_dir == Path("benchmarks/docling_eval").resolve():
        raise ValueError("Refusing to clean benchmarks/docling_eval directly.")

    shutil.rmtree(work_dir, ignore_errors=True)


def _official_result_path(work_dir: Path) -> Path:
    """
    Return the official docling-eval result JSON path for DocLayNetV1 layout.
    """
    return (
        work_dir
        / "results"
        / "evaluations"
        / "layout"
        / "evaluation_DocLayNetV1_layout.json"
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


def _stats_mean(payload: dict[str, Any], key: str) -> float:
    """
    Safely read '<metric>_stats.mean' from docling-eval output.

    docling-eval uses nested metric blocks such as:
        map_50_stats: {"mean": ...}
    """
    value = payload.get(key, {}).get("mean", -1.0)
    return float(value)


def _write_compact_summary(
    *,
    summary: OfficialDocLayNetSummary,
) -> None:
    """
    Save a small human-readable summary next to the official result.

    This does not replace the official docling-eval JSON.
    It only makes quick inspection easier.
    """
    payload = {
        "benchmark": "DocLayNetV1",
        "engine": "torvex_extract",
        "limit": summary.limit,
        "device": summary.device,
        "work_dir": str(summary.work_dir),
        "evaluated_samples": summary.evaluated_samples,
        "mAP": summary.map_score,
        "map_50_mean": summary.map_50_mean,
        "map_75_mean": summary.map_75_mean,
        "map_mean": summary.map_mean,
        "weighted_map_50_mean": summary.weighted_map_50_mean,
        "rejected_samples": summary.rejected_samples,
        "true_labels": summary.true_labels,
        "pred_labels": summary.pred_labels,
        "intersecting_labels": summary.intersecting_labels,
        "evaluations_per_class": summary.evaluations_per_class,
        "prediction_summary": {
            "requested": summary.prediction_summary.requested,
            "processed": summary.prediction_summary.processed,
            "predictions_written": summary.prediction_summary.predictions_written,
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


def run_official_doclaynet(
    *,
    limit: int,
    work_dir: Path = DEFAULT_WORK_DIR,
    clean: bool = True,
    save_normalized: bool = False,
    device: str = "cpu",
) -> OfficialDocLayNetSummary:
    """
    Run the full official DocLayNetV1 layout benchmark for Torvex.

    Args:
        limit:
            Number of DocLayNetV1 test samples to evaluate.
            Example: 25, 100, 1000.

        work_dir:
            Isolated generated-artifact folder.
            Default:
                benchmarks/docling_eval/DocLayNetV1_dev

        clean:
            If True, delete work_dir before running.
            This prevents stale GT/eval/results/predictions from mixing across runs.

        save_normalized:
            Debug option.
            If True, save normalized Torvex JSON next to prediction artifacts.
            If False, only prediction JSONs and official docling-eval artifacts are saved.

        device:
            Torvex Extract ONNX inference device.
            Supported:
                "cpu"
                "gpu"

    Returns:
        OfficialDocLayNetSummary with clean metrics and paths.
    """
    if limit <= 0:
        raise ValueError("limit must be a positive integer")

    if device not in {"cpu", "gpu"}:
        raise ValueError("device must be 'cpu' or 'gpu'")

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
            "DocLayNetV1",
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
    prediction_summary = generate_doclaynet_predictions(
        gt_dir=gt_test_dir,
        prediction_dir=prediction_dir,
        limit=limit,
        overwrite=True,
        save_normalized=save_normalized,
        device=device,
    )

    # 3. Build docling-eval evaluation dataset using File provider.
    _run_command(
        [
            "docling-eval",
            "create-eval",
            "--benchmark",
            "DocLayNetV1",
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
            "--end-index",
            str(limit),
            "--no-do-visualization",
        ]
    )

    # 4. Run official layout evaluation.
    _run_command(
        [
            "docling-eval",
            "evaluate",
            "--benchmark",
            "DocLayNetV1",
            "--modality",
            "layout",
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

    summary = OfficialDocLayNetSummary(
        limit=limit,
        device=device,
        work_dir=work_dir,
        prediction_summary=prediction_summary,
        evaluated_samples=int(official_result.get("evaluated_samples", 0)),
        map_score=float(official_result.get("mAP", -1.0)),
        map_50_mean=_stats_mean(official_result, "map_50_stats"),
        map_75_mean=_stats_mean(official_result, "map_75_stats"),
        map_mean=_stats_mean(official_result, "map_stats"),
        weighted_map_50_mean=_stats_mean(
            official_result,
            "weighted_map_50_stats",
        ),
        rejected_samples=dict(official_result.get("rejected_samples", {})),
        true_labels=dict(official_result.get("true_labels", {})),
        pred_labels=dict(official_result.get("pred_labels", {})),
        intersecting_labels=list(official_result.get("intersecting_labels", [])),
        evaluations_per_class=list(
            official_result.get("evaluations_per_class", [])
        ),
        official_result_path=official_result_path,
        summary_path=summary_path,
    )

    _write_compact_summary(summary=summary)

    return summary