"""
Official OmniDocBench evaluator wrapper.

Purpose
-------
Run the official OmniDocBench end-to-end evaluator on Torvex predictions.

Locked benchmark decision
-------------------------
OmniDocBench is used as a scanned/image-page benchmark only.

We do not use:
    - ori_pdfs/
    - digital PDF mode
    - formula CDM
    - COCO Det mAP in this path

This wrapper evaluates:
    - text_block     -> Edit_dist
    - table          -> TEDS
    - reading_order  -> Edit_dist

Flow
----
1. Generate Torvex .md predictions through harnesses/omnidocbench_eval.py.
2. Write subset GT JSON matching exactly the selected samples.
3. Write official OmniDocBench config.yaml.
4. Call:

       omnidocbench-eval --config config.yaml

5. Read official result JSON from:

       work_dir/result/torvex_extract_quick_match_metric_result.json

No metric formula is implemented here. Metrics are computed by omnidocbench-eval.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from torvex_bench.datasets.omnidocbench import (
    OmniDocBenchSample,
    iter_omnidocbench_samples_from_manifest,
    prepare_omnidocbench,
)
from torvex_bench.harnesses.omnidocbench_eval import (
    OmniDocBenchPredictionSummary,
    generate_omnidocbench_predictions,
)


DEFAULT_WORK_DIR = Path("benchmarks/omnidocbench/OmniDocBench_scanned")
DEFAULT_ENGINE_NAME = "torvex_extract"


@dataclass(slots=True)
class OfficialOmniDocBenchSummary:
    """Compact summary for one official OmniDocBench run."""

    dataset: str
    engine: str
    limit: int
    work_dir: Path
    gt_subset_path: Path
    predictions_dir: Path
    config_path: Path
    official_result_path: Path
    official_run_summary_path: Path
    summary_path: Path
    eval_returncode: int
    prediction_summary: dict[str, Any]
    metrics: dict[str, Any]


def resolve_omnidocbench_eval_bin(eval_bin: str | Path | None = None) -> Path:
    """
    Resolve the omnidocbench-eval executable.

    Priority:
        1. explicit eval_bin argument
        2. OMNIDOCBENCH_EVAL_BIN env var
        3. local isolated venv path used in this repo

    Windows default:
        data/venvs/omnidocbench/Scripts/omnidocbench-eval.exe

    Linux/Kaggle default:
        data/venvs/omnidocbench/bin/omnidocbench-eval
    """
    if eval_bin:
        return Path(eval_bin)

    env_value = os.getenv("OMNIDOCBENCH_EVAL_BIN")
    if env_value:
        return Path(env_value)

    windows_path = Path("data/venvs/omnidocbench/Scripts/omnidocbench-eval.exe")
    if windows_path.exists():
        return windows_path

    return Path("data/venvs/omnidocbench/bin/omnidocbench-eval")


def _json_safe_path(path: Path) -> str:
    """Return path as a stable JSON/YAML-safe POSIX string."""
    return path.resolve().as_posix()


def write_subset_gt_json(
    samples: list[OmniDocBenchSample],
    output_path: str | Path,
) -> Path:
    """
    Write a GT JSON containing only selected samples.

    This is important for smoke runs.

    If we pass full OmniDocBench.json while only writing 1 or 3 predictions,
    official OmniDocBench will evaluate all missing pages as empty predictions.
    So for limit=N, we write a subset GT with exactly N page records.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for sample in samples:
        records.append(
            {
                "page_info": sample.page_info,
                "layout_dets": sample.layout_dets,
                "extra": sample.extra,
            }
        )

    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return output_path


def write_official_omnidocbench_config(
    *,
    config_path: str | Path,
    gt_json_path: str | Path,
    predictions_dir: str | Path,
    match_workers: int = 1,
    teds_workers: int = 1,
) -> Path:
    """
    Write official omnidocbench-eval config.yaml.

    Formula CDM is intentionally omitted.

    COCO Det mAP is intentionally not included here because this is the
    end-to-end Markdown evaluation path, not the separate layout detection path.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    gt_path = _json_safe_path(Path(gt_json_path))
    pred_path = _json_safe_path(Path(predictions_dir))

    config_text = f"""end2end_eval:
  metrics:
    text_block:
      metric:
        - Edit_dist
    table:
      metric:
        - TEDS
      teds_workers: {int(teds_workers)}
    reading_order:
      metric:
        - Edit_dist
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: "{gt_path}"
    prediction:
      data_path: "{pred_path}"
    match_method: quick_match
    match_workers: {int(match_workers)}
    quick_match_truncated_timeout_sec: 300
    match_timeout_sec: 420
    timeout_fallback_max_chunk_span: 10
    timeout_fallback_order_penalty: 0.10
"""

    config_path.write_text(config_text, encoding="utf-8")
    return config_path


def run_official_omnidocbench_eval(
    *,
    config_path: str | Path,
    work_dir: str | Path,
    eval_bin: str | Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """
    Call the official OmniDocBench evaluator.

    Important:
        cwd=work_dir

    The official evaluator writes ./result relative to cwd.
    """
    resolved_eval_bin = resolve_omnidocbench_eval_bin(eval_bin)

    if not resolved_eval_bin.exists():
        raise FileNotFoundError(
            f"omnidocbench-eval executable not found: {resolved_eval_bin}. "
            "Create the isolated env first or set OMNIDOCBENCH_EVAL_BIN."
        )

    command = [
        str(resolved_eval_bin),
        "--config",
        str(Path(config_path).resolve()),
    ]

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")

    return subprocess.run(
        command,
        cwd=Path(work_dir),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        env=env,
    )


def read_official_metrics(result_path: str | Path) -> dict[str, Any]:
    """Read official metric result JSON if present."""
    result_path = Path(result_path)

    if not result_path.exists():
        return {}

    return json.loads(result_path.read_text(encoding="utf-8"))


def clean_omnidocbench_work_dir(work_dir: str | Path) -> None:
    """
    Clean generated OmniDocBench run artifacts while keeping gt_dataset.

    We keep:
        gt_dataset/
            OmniDocBench.json
            images/
            sample_manifest.jsonl

    We remove:
        predictions/
        temp_pdfs/
        raw_outputs/
        normalized/
        result/
        config.yaml
        summary.json
        omnidocbench_eval_stdout.log
        omnidocbench_eval_stderr.log
    """
    work_dir = Path(work_dir)

    for child_name in [
        "predictions",
        "temp_pdfs",
        "raw_outputs",
        "normalized",
        "result",
    ]:
        child = work_dir / child_name
        if child.exists():
            shutil.rmtree(child)

    for file_name in [
        "config.yaml",
        "summary.json",
        "omnidocbench_eval_stdout.log",
        "omnidocbench_eval_stderr.log",
    ]:
        path = work_dir / file_name
        if path.exists():
            path.unlink()


def run_official_omnidocbench(
    *,
    work_dir: Path = DEFAULT_WORK_DIR,
    limit: int = 3,
    clean: bool = True,
    save_normalized: bool = False,
    device: str = "cpu",
    eval_bin: str | Path | None = None,
) -> OfficialOmniDocBenchSummary:
    """
    Generate Torvex predictions and run official OmniDocBench evaluation.

    Command-style behavior:
        - prepares first limit Omni samples
        - generates .md predictions
        - writes subset GT JSON
        - writes official config.yaml
        - calls omnidocbench-eval
        - writes compact summary.json

    This is the function the CLI will call.
    """
    work_dir = Path(work_dir)

    if clean:
        clean_omnidocbench_work_dir(work_dir)

    gt_dir = work_dir / "gt_dataset"
    predictions_dir = work_dir / "predictions" / DEFAULT_ENGINE_NAME
    gt_subset_path = gt_dir / f"OmniDocBench_subset_limit_{limit}.json"
    config_path = work_dir / "config.yaml"
    summary_path = work_dir / "summary.json"

    prediction_summary: OmniDocBenchPredictionSummary = generate_omnidocbench_predictions(
    work_dir=work_dir,
    limit=limit,
    overwrite=clean,
    save_raw=False,
    save_normalized=save_normalized,
    device=device,
)
    manifest_path = prepare_omnidocbench(
        raw_data_dir=gt_dir,
        output_dir=gt_dir,
        limit=limit,
        download_images=True,
    )
    samples = iter_omnidocbench_samples_from_manifest(manifest_path, limit=limit)

    write_subset_gt_json(samples, gt_subset_path)

    write_official_omnidocbench_config(
        config_path=config_path,
        gt_json_path=gt_subset_path,
        predictions_dir=predictions_dir,
        match_workers=1,
        teds_workers=1,
    )

    completed = run_official_omnidocbench_eval(
        config_path=config_path,
        work_dir=work_dir,
        eval_bin=eval_bin,
    )

    save_name = f"{predictions_dir.name}_quick_match"
    official_result_path = work_dir / "result" / f"{save_name}_metric_result.json"
    official_run_summary_path = work_dir / "result" / f"{save_name}_run_summary.json"

    metrics = read_official_metrics(official_result_path)

    summary = OfficialOmniDocBenchSummary(
        dataset="OmniDocBench_scanned",
        engine=DEFAULT_ENGINE_NAME,
        limit=limit,
        work_dir=work_dir,
        gt_subset_path=gt_subset_path,
        predictions_dir=predictions_dir,
        config_path=config_path,
        official_result_path=official_result_path,
        official_run_summary_path=official_run_summary_path,
        summary_path=summary_path,
        eval_returncode=int(completed.returncode),
        prediction_summary=asdict(prediction_summary),
        metrics=metrics,
    )

    summary_path.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    if completed.returncode != 0:
        stdout_path = work_dir / "omnidocbench_eval_stdout.log"
        stderr_path = work_dir / "omnidocbench_eval_stderr.log"
        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")

        raise RuntimeError(
            f"omnidocbench-eval failed with returncode={completed.returncode}. "
            f"See {stdout_path} and {stderr_path}."
        )

    return summary