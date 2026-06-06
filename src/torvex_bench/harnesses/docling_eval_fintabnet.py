from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
from PIL import Image

from torvex_bench.adapters.torvex_extract_adapter import TorvexExtractAdapter
from torvex_bench.exporters.docling_eval import export_rows_as_docling_json
from torvex_bench.normalizer import normalize_document


@dataclass(frozen=True)
class FinTabNetPredictionSummary:
    requested: int
    processed: int
    predictions_written: int
    missing_tables: int
    skipped_existing: int
    errors: int
    prediction_dir: Path
    normalized_dir: Path | None


def iter_fintabnet_gt_rows(
    gt_dir: Path,
    *,
    limit: int | None = None,
    start_index: int = 0,
) -> Iterator[dict]:
    """
    Iterate official docling-eval FinTabNet GT parquet rows.

    The rows already contain:
    - document_id
    - BinaryDocument image bytes
    - GroundTruthDocument
    """
    seen = 0
    yielded = 0

    for parquet_path in sorted(gt_dir.glob("*.parquet")):
        table = pq.read_table(parquet_path)

        for row in table.to_pylist():
            if seen < start_index:
                seen += 1
                continue

            if limit is not None and yielded >= limit:
                return

            yield row
            yielded += 1
            seen += 1


def image_bytes_to_pdf(image_bytes: bytes, pdf_path: Path) -> None:
    """
    Convert FinTabNet PNG bytes into one-page PDF for Torvex Extract.

    Production default uses a TemporaryDirectory, so PNG/PDF inputs are not kept.
    """
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    image_path = pdf_path.with_suffix(".png")
    image_path.write_bytes(image_bytes)

    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(pdf_path, "PDF", resolution=72.0)


def extract_first_table_rows(normalized: dict) -> list[list[str]]:
    """
    FinTabNet has one table crop per sample.
    Use the first extracted table from the normalized Torvex document.
    """
    for page in normalized.get("pages", []):
        tables = page.get("tables", []) or []
        if tables:
            rows = tables[0].get("rows", []) or []
            if rows:
                return rows

    tables = normalized.get("tables", []) or []
    if tables:
        rows = tables[0].get("rows", []) or []
        if rows:
            return rows

    return []


def generate_fintabnet_predictions(
    *,
    gt_dir: Path,
    prediction_dir: Path,
    limit: int | None = None,
    start_index: int = 0,
    overwrite: bool = False,
    save_normalized: bool = False,
    normalized_dir: Path | None = None,
    keep_inputs: bool = False,
    inputs_dir: Path | None = None,
) -> FinTabNetPredictionSummary:
    """
    Generate docling-eval File-provider predictions for FinTabNet.

    Official flow:
      GT parquet PNG bytes -> temporary one-page PDF -> Torvex -> normalized
      -> DoclingDocument JSON prediction at <document_id>.json
    """
    prediction_dir.mkdir(parents=True, exist_ok=True)

    if save_normalized:
        normalized_dir = normalized_dir or Path(
            "results/raw/fintabnet/torvex_extract/normalized"
        )
        normalized_dir.mkdir(parents=True, exist_ok=True)

    if keep_inputs:
        inputs_dir = inputs_dir or Path("results/raw/fintabnet/torvex_extract/inputs")
        inputs_dir.mkdir(parents=True, exist_ok=True)

    adapter = TorvexExtractAdapter()

    rows = list(
        iter_fintabnet_gt_rows(
            gt_dir,
            limit=limit,
            start_index=start_index,
        )
    )

    requested = len(rows)
    processed = 0
    predictions_written = 0
    missing_tables = 0
    skipped_existing = 0
    errors = 0

    for row in rows:
        doc_id = row["document_id"]
        prediction_path = prediction_dir / f"{doc_id}.json"

        if prediction_path.exists() and not overwrite:
            skipped_existing += 1
            continue

        try:
            if keep_inputs:
                assert inputs_dir is not None
                pdf_path = inputs_dir / f"{doc_id}.pdf"
                image_bytes_to_pdf(row["BinaryDocument"], pdf_path)
                document_result = adapter.extract_document(pdf_path)
            else:
                with tempfile.TemporaryDirectory(prefix="torvex_fintabnet_") as tmpdir:
                    pdf_path = Path(tmpdir) / f"{doc_id}.pdf"
                    image_bytes_to_pdf(row["BinaryDocument"], pdf_path)
                    document_result = adapter.extract_document(pdf_path)

            normalized = normalize_document(document_result)

            if save_normalized:
                assert normalized_dir is not None
                normalized_path = normalized_dir / f"{doc_id}.json"
                normalized_path.write_text(
                    json.dumps(normalized, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            table_rows = extract_first_table_rows(normalized)

            if not table_rows:
                missing_tables += 1

            # Always write a prediction document.
            # If no table is found, this writes an empty DoclingDocument instead of
            # causing a missing prediction file.
            export_rows_as_docling_json(
                rows=table_rows,
                output_path=prediction_path,
                name=f"torvex prediction {doc_id}",
            )

            predictions_written += 1
            processed += 1

        except Exception as exc:
            errors += 1
            print(f"[ERROR] {doc_id}: {exc}")

    return FinTabNetPredictionSummary(
        requested=requested,
        processed=processed,
        predictions_written=predictions_written,
        missing_tables=missing_tables,
        skipped_existing=skipped_existing,
        errors=errors,
        prediction_dir=prediction_dir,
        normalized_dir=normalized_dir if save_normalized else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("benchmarks/docling_eval/FinTabNet/gt_dataset/test"),
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=Path("benchmarks/docling_eval/FinTabNet/predictions/torvex_extract"),
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-normalized", action="store_true")
    parser.add_argument("--keep-inputs", action="store_true")

    args = parser.parse_args()

    summary = generate_fintabnet_predictions(
        gt_dir=args.gt_dir,
        prediction_dir=args.prediction_dir,
        limit=args.limit,
        start_index=args.start_index,
        overwrite=args.overwrite,
        save_normalized=args.save_normalized,
        keep_inputs=args.keep_inputs,
    )

    print("requested=", summary.requested)
    print("processed=", summary.processed)
    print("predictions_written=", summary.predictions_written)
    print("missing_tables=", summary.missing_tables)
    print("skipped_existing=", summary.skipped_existing)
    print("errors=", summary.errors)
    print("prediction_dir=", summary.prediction_dir)
    print("normalized_dir=", summary.normalized_dir)


if __name__ == "__main__":
    main()