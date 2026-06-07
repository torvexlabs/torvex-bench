"""
OmniDocBench prediction harness.

Purpose
-------
Generate official OmniDocBench end-to-end Markdown predictions from Torvex
Extract output.

This module does prediction generation only.

Flow:
    1. Read prepared OmniDocBench sample_manifest.jsonl.
    2. For each page image, create a temporary one-page scanned PDF.
    3. Run TorvexExtractAdapter on that PDF.
    4. Normalize DocumentResult.
    5. Export <image_stem>.md for official omnidocbench-eval.

It does NOT:
    - compute NED
    - compute TEDS
    - compute reading-order score
    - call omnidocbench-eval
    - use ori_pdfs
    - use pdfs/

Official evaluator wrapper comes later in:

    harnesses/official_omnidocbench.py
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from torvex_bench.adapters.base import DocumentResult
from torvex_bench.adapters.torvex_extract_adapter import TorvexExtractAdapter
from torvex_bench.datasets.omnidocbench import (
    OmniDocBenchSample,
    iter_omnidocbench_samples_from_manifest,
    prepare_omnidocbench,
)
from torvex_bench.exporters.omnidocbench_markdown import (
    export_sample_markdown_prediction,
)
from torvex_bench.normalizer import normalize_document


class SupportsExtractDocument(Protocol):
    """Small protocol so tests can inject a fake adapter."""

    def extract_document(self, pdf_path: str | Path) -> DocumentResult:
        """Extract one PDF and return DocumentResult."""
        ...


@dataclass(slots=True)
class OmniDocBenchPredictionSummary:
    """
    Summary for one OmniDocBench prediction generation run.

    requested:
        Number of samples loaded from manifest.

    processed:
        Number of samples attempted, excluding skipped existing predictions.

    predictions_written:
        Number of non-empty/normal prediction files written after successful extraction.

    empty_predictions_written:
        Number of empty .md files written after extraction errors.
        These are scored honestly by the official evaluator as bad predictions.

    skipped_existing:
        Existing .md prediction files skipped when overwrite=False.

    errors:
        Number of extraction/export errors.

    prediction_dir:
        Folder containing official .md predictions.

    temp_pdfs_dir:
        Folder containing image-derived temporary PDFs.

    raw_dir / normalized_dir:
        Optional debug artifact folders.
    """

    requested: int
    processed: int
    predictions_written: int
    empty_predictions_written: int
    skipped_existing: int
    errors: int
    prediction_dir: Path
    temp_pdfs_dir: Path
    raw_dir: Path | None = None
    normalized_dir: Path | None = None


def image_to_scanned_pdf(image_path: str | Path, pdf_path: str | Path) -> Path:
    """
    Convert one OmniDocBench page image into a one-page scanned PDF.

    Why this exists:
        OmniDocBench current-main official source is image pages.
        Torvex Extract expects PDF input.
        So this is the input bridge, same idea as the FinTabNet image-to-PDF bridge.

    This does not create a digital/text-layer PDF.
    It creates an image-only PDF, which keeps the benchmark as scanned/OCR path.
    """
    image_path = Path(image_path)
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(pdf_path, "PDF", resolution=72.0)

    return pdf_path


def _write_json(path: Path, payload: dict | list) -> None:
    """Write JSON artifact with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_empty_prediction(prediction_path: Path) -> None:
    """
    Write an empty prediction file.

    Missing predictions are evaluated as empty pages by official OmniDocBench.
    We write the empty file explicitly so the prediction folder is complete and
    the failure is visible in torvex-bench summary counts.
    """
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text("", encoding="utf-8")


def generate_omnidocbench_predictions_from_samples(
    *,
    samples: list[OmniDocBenchSample],
    prediction_dir: Path,
    temp_pdfs_dir: Path,
    adapter: SupportsExtractDocument,
    overwrite: bool = False,
    save_raw: bool = False,
    raw_dir: Path | None = None,
    save_normalized: bool = False,
    normalized_dir: Path | None = None,
    write_empty_prediction_on_error: bool = True,
) -> OmniDocBenchPredictionSummary:
    """
    Generate OmniDocBench .md predictions for already-materialized samples.

    This function is test-friendly because the adapter is injected.

    Normal production path should call generate_omnidocbench_predictions(),
    which prepares the dataset and creates TorvexExtractAdapter.
    """
    prediction_dir.mkdir(parents=True, exist_ok=True)
    temp_pdfs_dir.mkdir(parents=True, exist_ok=True)

    if save_raw:
        raw_dir = raw_dir or prediction_dir.parent / "raw_outputs"
        raw_dir.mkdir(parents=True, exist_ok=True)

    if save_normalized:
        normalized_dir = normalized_dir or prediction_dir.parent / "normalized"
        normalized_dir.mkdir(parents=True, exist_ok=True)

    requested = len(samples)
    processed = 0
    predictions_written = 0
    empty_predictions_written = 0
    skipped_existing = 0
    errors = 0

    for sample in samples:
        prediction_path = prediction_dir / sample.prediction_filename

        if prediction_path.exists() and not overwrite:
            skipped_existing += 1
            continue

        processed += 1

        try:
            pdf_path = temp_pdfs_dir / f"{sample.image_stem}.pdf"
            image_to_scanned_pdf(sample.image_path, pdf_path)

            document = adapter.extract_document(pdf_path)
            normalized = normalize_document(document)

            if save_raw and raw_dir is not None:
                _write_json(raw_dir / f"{sample.image_stem}.json", asdict(document))

            if save_normalized and normalized_dir is not None:
                _write_json(normalized_dir / f"{sample.image_stem}.json", normalized)

            export_sample_markdown_prediction(
                normalized,
                prediction_filename=sample.prediction_filename,
                predictions_dir=prediction_dir,
            )

            predictions_written += 1

        except Exception as exc:
            errors += 1
            print(f"[ERROR] {sample.sample_id} {sample.image_filename}: {exc}")

            if save_raw and raw_dir is not None:
                _write_json(
                    raw_dir / f"{sample.image_stem}.error.json",
                    {
                        "sample_id": sample.sample_id,
                        "image_filename": sample.image_filename,
                        "error": str(exc),
                    },
                )

            if write_empty_prediction_on_error:
                _write_empty_prediction(prediction_path)
                empty_predictions_written += 1

    return OmniDocBenchPredictionSummary(
        requested=requested,
        processed=processed,
        predictions_written=predictions_written,
        empty_predictions_written=empty_predictions_written,
        skipped_existing=skipped_existing,
        errors=errors,
        prediction_dir=prediction_dir,
        temp_pdfs_dir=temp_pdfs_dir,
        raw_dir=raw_dir if save_raw else None,
        normalized_dir=normalized_dir if save_normalized else None,
    )


def generate_omnidocbench_predictions(
    *,
    work_dir: Path = Path("benchmarks/omnidocbench/OmniDocBench_scanned"),
    limit: int = 3,
    overwrite: bool = False,
    save_raw: bool = False,
    save_normalized: bool = False,
    device: str = "cpu",
) -> OmniDocBenchPredictionSummary:
    """
    Prepare OmniDocBench samples and generate Torvex Markdown predictions.

    Folder layout:
        work_dir/
          gt_dataset/
            OmniDocBench.json
            images/
            sample_manifest.jsonl

          predictions/
            torvex_extract/
              <image_stem>.md

          temp_pdfs/
            torvex_extract/
              <image_stem>.pdf

          raw_outputs/
            torvex_extract/
              <image_stem>.json

          normalized/
            torvex_extract/
              <image_stem>.json

    Command-style behavior:
        - limit=1 downloads/uses first sample image only.
        - limit=3 downloads/uses first 3 images only.
        - existing images are reused.
        - existing predictions are skipped unless overwrite=True.
    """
    gt_dir = work_dir / "gt_dataset"
    prediction_dir = work_dir / "predictions" / "torvex_extract"
    temp_pdfs_dir = work_dir / "temp_pdfs" / "torvex_extract"
    raw_dir = work_dir / "raw_outputs" / "torvex_extract"
    normalized_dir = work_dir / "normalized" / "torvex_extract"

    manifest_path = prepare_omnidocbench(
        raw_data_dir=gt_dir,
        output_dir=gt_dir,
        limit=limit,
        download_images=True,
    )

    samples = iter_omnidocbench_samples_from_manifest(manifest_path, limit=limit)

    adapter = TorvexExtractAdapter(device=device)

    return generate_omnidocbench_predictions_from_samples(
        samples=samples,
        prediction_dir=prediction_dir,
        temp_pdfs_dir=temp_pdfs_dir,
        adapter=adapter,
        overwrite=overwrite,
        save_raw=save_raw,
        raw_dir=raw_dir,
        save_normalized=save_normalized,
        normalized_dir=normalized_dir,
    )