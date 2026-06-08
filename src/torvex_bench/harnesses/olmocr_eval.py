"""
olmOCR-Bench prediction harness.

Purpose
-------
Generate official olmOCR-Bench Markdown predictions from Torvex Extract output.

This module does prediction generation only.

Flow:
    1. Prepare/load olmOCR-Bench manifest.
    2. For each selected PDF, run TorvexExtractAdapter.
    3. Normalize DocumentResult.
    4. Export one .md prediction per tested page:
       <bench_data>/torvex_extract/<pdf_stem>_pg<page>_repeat1.md

It does NOT:
    - call python -m olmocr.bench.benchmark
    - compute pass rates
    - parse official evaluator output

Official evaluator wrapper comes later in:
    harnesses/official_olmocr.py
"""

from __future__ import annotations

import json
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from torvex_bench.adapters.base import DocumentResult
from torvex_bench.adapters.torvex_extract_adapter import TorvexExtractAdapter
from torvex_bench.datasets.olmocr import (
    OlmOCRBenchSample,
    bench_data_dir,
    iter_olmocr_samples_from_manifest,
    prepare_olmocr_bench,
)
from torvex_bench.exporters.olmocr_markdown import export_olmocr_markdown_prediction
from torvex_bench.normalizer import normalize_document


DEFAULT_ENGINE_NAME = "torvex_extract"


class SupportsExtractDocument(Protocol):
    """Small protocol so tests can inject a fake adapter."""

    def extract_document(self, pdf_path: str | Path) -> DocumentResult:
        """Extract one PDF and return DocumentResult."""
        ...


@dataclass(slots=True)
class OlmOCRPredictionSummary:
    requested: int
    processed: int
    predictions_written: int
    empty_predictions_written: int
    skipped_existing: int
    errors: int
    prediction_dir: Path
    normalized_dir: Path | None = None
    raw_dir: Path | None = None


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _write_empty_prediction(prediction_path: Path) -> None:
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text("", encoding="utf-8")


def prediction_path_for_sample_page(
    *,
    sample: OlmOCRBenchSample,
    prediction_dir: Path,
    page: int,
    repeat: int = 1,
) -> Path:
    """
    Return official olmOCR-Bench prediction path for one sample/page.

    Example:
        prediction_dir / "old_scans/1_pg1_repeat1.md"
    """
    return prediction_dir / sample.prediction_filename_for_page(page, repeat=repeat)


def generate_olmocr_predictions_from_samples(
    *,
    samples: list[OlmOCRBenchSample],
    prediction_dir: Path,
    adapter: SupportsExtractDocument,
    overwrite: bool = False,
    save_raw: bool = False,
    raw_dir: Path | None = None,
    save_normalized: bool = False,
    normalized_dir: Path | None = None,
    write_empty_prediction_on_error: bool = True,
) -> OlmOCRPredictionSummary:
    """
    Generate .md predictions for already-prepared olmOCR-Bench samples.

    This function is test-friendly because the adapter is injected.
    """
    prediction_dir.mkdir(parents=True, exist_ok=True)
    error_dir = prediction_dir.parent / "errors" / DEFAULT_ENGINE_NAME
    error_dir.mkdir(parents=True, exist_ok=True)

    if save_raw:
        raw_dir = raw_dir or prediction_dir.parent / "raw_outputs" / DEFAULT_ENGINE_NAME
        raw_dir.mkdir(parents=True, exist_ok=True)

    if save_normalized:
        normalized_dir = normalized_dir or prediction_dir.parent / "normalized" / DEFAULT_ENGINE_NAME
        normalized_dir.mkdir(parents=True, exist_ok=True)

    requested = len(samples)
    processed = 0
    predictions_written = 0
    empty_predictions_written = 0
    skipped_existing = 0
    errors = 0

    for sample in samples:
        pages = sample.pages or [1]
        prediction_paths = [
            prediction_path_for_sample_page(
                sample=sample,
                prediction_dir=prediction_dir,
                page=page,
            )
            for page in pages
        ]

        if all(path.exists() for path in prediction_paths) and not overwrite:
            skipped_existing += 1
            continue

        processed += 1

        try:
            document = adapter.extract_document(sample.local_pdf_path)
            normalized = normalize_document(document)

            if save_raw and raw_dir is not None:
                _write_json(raw_dir / f"{sample.sample_id}.json", asdict(document))

            if save_normalized and normalized_dir is not None:
                _write_json(normalized_dir / f"{sample.sample_id}.json", normalized)

            for page, prediction_path in zip(pages, prediction_paths, strict=True):
                if prediction_path.exists() and not overwrite:
                    continue

                export_olmocr_markdown_prediction(
                    normalized,
                    prediction_path,
                    page=page,
                )
                predictions_written += 1

        except Exception as exc:
            errors += 1
            traceback_text = traceback.format_exc()

            print(f"[ERROR] {sample.sample_id} {sample.pdf}: {exc}")

            _write_json(
                error_dir / f"{sample.sample_id}.error.json",
                {
                    "sample_id": sample.sample_id,
                    "pdf": sample.pdf,
                    "local_pdf_path": str(sample.local_pdf_path),
                    "pages": sample.pages,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback_text,
                },
            )

            if write_empty_prediction_on_error:
                for prediction_path in prediction_paths:
                    _write_empty_prediction(prediction_path)
                    empty_predictions_written += 1

    return OlmOCRPredictionSummary(
        requested=requested,
        processed=processed,
        predictions_written=predictions_written,
        empty_predictions_written=empty_predictions_written,
        skipped_existing=skipped_existing,
        errors=errors,
        prediction_dir=prediction_dir,
        normalized_dir=normalized_dir if save_normalized else None,
        raw_dir=raw_dir if save_raw else None,
    )


def generate_olmocr_predictions(
    *,
    work_dir: Path = Path("benchmarks/olmocr/olmOCR_Bench_non_math"),
    limit: int = 3,
    track: str = "non_math",
    overwrite: bool = False,
    save_raw: bool = False,
    save_normalized: bool = False,
    device: str = "cpu",
) -> OlmOCRPredictionSummary:
    """
    Prepare olmOCR-Bench samples and generate Torvex Markdown predictions.

    Folder layout:
        work_dir/
          bench_data/
            *.jsonl
            pdfs/
            sample_manifest.jsonl
            torvex_extract/
              <pdf_stem>_pg1_repeat1.md
            normalized/
            raw_outputs/
            errors/
    """
    manifest_path = prepare_olmocr_bench(
        work_dir=work_dir,
        limit=limit,
        track=track,
        download_pdfs=True,
    )

    samples = iter_olmocr_samples_from_manifest(manifest_path, limit=limit)

    data_dir = bench_data_dir(work_dir)
    prediction_dir = data_dir / DEFAULT_ENGINE_NAME
    raw_dir = data_dir / "raw_outputs" / DEFAULT_ENGINE_NAME
    normalized_dir = data_dir / "normalized" / DEFAULT_ENGINE_NAME

    adapter = TorvexExtractAdapter(device=device)

    return generate_olmocr_predictions_from_samples(
        samples=samples,
        prediction_dir=prediction_dir,
        adapter=adapter,
        overwrite=overwrite,
        save_raw=save_raw,
        raw_dir=raw_dir,
        save_normalized=save_normalized,
        normalized_dir=normalized_dir,
    )