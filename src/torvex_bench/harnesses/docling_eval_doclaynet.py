from __future__ import annotations

"""
DocLayNetV1 prediction harness for official docling-eval layout scoring.

This module generates DoclingDocument JSON predictions for DocLayNetV1
using Torvex Extract.

High-level flow:
1. Read official docling-eval DocLayNetV1 GT parquet rows.
2. For each row, write BinaryDocument PDF bytes to a temporary PDF.
3. Run TorvexExtractAdapter on that PDF.
4. Normalize the DocumentResult.
5. Convert Torvex layout zones into asset-free DoclingDocument JSON.
6. Save prediction as:
   <prediction_dir>/<document_id>.json

Important:
- This module does not compute mAP.
- docling-eval remains the official scorer.
- Prediction JSON must not include page image or picture image URIs.
"""

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

from torvex_bench.adapters.torvex_extract_adapter import TorvexExtractAdapter
from torvex_bench.normalizer import normalize_document


DOCLAYNET_V1_LABELS = {
    "caption",
    "footnote",
    "formula",
    "list_item",
    "page_footer",
    "page_header",
    "picture",
    "section_header",
    "table",
    "text",
    "title",
}


ZONE_TYPE_TO_DOCLAYNET_LABEL = {
    # Text-like
    "abstract": "text",
    "algorithm": "text",
    "aside_text": "text",
    "content": "text",
    "number": "text",
    "reference": "text",
    "reference_content": "text",
    "text": "text",
    "vertical_text": "text",

    # Titles / headings
    "doc_title": "title",
    "paragraph_title": "section_header",
    "figure_title": "caption",

    # Page furniture
    "header": "page_header",
    "footer": "page_footer",
    "footnote": "footnote",
    "vision_footnote": "footnote",

    # Layout objects
    "table": "table",
    "image": "picture",
    "chart": "picture",
    "seal": "picture",
    "header_image": "picture",
    "footer_image": "picture",

    # Formula detection only; no LaTeX/content extraction.
    "display_formula": "formula",
    "inline_formula": "formula",
    "formula_number": "formula",
}


@dataclass(frozen=True)
class DocLayNetPredictionSummary:
    requested: int
    processed: int
    predictions_written: int
    skipped_existing: int
    errors: int
    prediction_dir: Path
    normalized_dir: Path | None


def iter_gt_rows(
    gt_dir: Path,
    *,
    limit: int,
    start_index: int = 0,
) -> Iterator[dict[str, Any]]:
    """
    Iterate rows from official docling-eval DocLayNetV1 GT parquet files.
    """
    parquet_files = sorted(gt_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {gt_dir}")

    yielded = 0
    seen = 0

    for parquet_path in parquet_files:
        rows = pq.read_table(parquet_path).to_pylist()

        for row in rows:
            if seen < start_index:
                seen += 1
                continue

            if yielded >= limit:
                return

            yielded += 1
            seen += 1
            yield row


def _page_size_from_gt(row: dict[str, Any]) -> tuple[float, float]:
    """
    Read page size from the official GroundTruthDocument JSON.
    Fallback to US Letter if unavailable.
    """
    try:
        gt_doc = json.loads(row["GroundTruthDocument"])
        page = gt_doc.get("pages", {}).get("1", {})
        size = page.get("size", {})
        width = float(size.get("width", 612.0))
        height = float(size.get("height", 792.0))
        return width, height
    except Exception:
        return 612.0, 792.0


def _bbox_from_pdfium(
    bbox_pdfium: Any,
    *,
    page_width: float,
    page_height: float,
) -> dict[str, Any] | None:
    """
    Convert Torvex bbox_pdfium [left, bottom, right, top]
    into DoclingDocument bbox dict with BOTTOMLEFT origin.
    """
    if bbox_pdfium is None:
        return None

    try:
        left, bottom, right, top = [float(value) for value in bbox_pdfium]
    except Exception:
        return None

    left = max(0.0, min(page_width, left))
    right = max(0.0, min(page_width, right))
    bottom = max(0.0, min(page_height, bottom))
    top = max(0.0, min(page_height, top))

    if right <= left or top <= bottom:
        return None

    return {
        "l": left,
        "t": top,
        "r": right,
        "b": bottom,
        "coord_origin": "BOTTOMLEFT",
    }


def _make_prov(
    *,
    page_no: int,
    bbox: dict[str, Any],
    text_len: int = 0,
) -> list[dict[str, Any]]:
    return [
        {
            "page_no": page_no,
            "bbox": bbox,
            "charspan": [0, max(0, int(text_len))],
        }
    ]


def _make_text_item(
    *,
    index: int,
    label: str,
    text: str,
    page_no: int,
    bbox: dict[str, Any],
) -> dict[str, Any]:
    return {
        "self_ref": f"#/texts/{index}",
        "parent": {"$ref": "#/body"},
        "children": [],
        "content_layer": "body",
        "label": label,
        "prov": _make_prov(page_no=page_no, bbox=bbox, text_len=len(text)),
        "orig": text,
        "text": text,
    }


def _make_picture_item(
    *,
    index: int,
    page_no: int,
    bbox: dict[str, Any],
) -> dict[str, Any]:
    return {
        "self_ref": f"#/pictures/{index}",
        "parent": {"$ref": "#/body"},
        "children": [],
        "content_layer": "body",
        "label": "picture",
        "prov": _make_prov(page_no=page_no, bbox=bbox),
        "captions": [],
        "references": [],
        "footnotes": [],
        "annotations": [],
    }


def _make_table_item(
    *,
    index: int,
    page_no: int,
    bbox: dict[str, Any],
) -> dict[str, Any]:
    return {
        "self_ref": f"#/tables/{index}",
        "parent": {"$ref": "#/body"},
        "children": [],
        "content_layer": "body",
        "label": "table",
        "prov": _make_prov(page_no=page_no, bbox=bbox),
        "captions": [],
        "references": [],
        "footnotes": [],
        "data": {
            "table_cells": [],
            "num_rows": 0,
            "num_cols": 0,
            "orientation": "rot_0",
            "grid": [],
        },
        "annotations": [],
    }


def _doclaynet_label_for_zone(zone_type: str) -> str | None:
    label = ZONE_TYPE_TO_DOCLAYNET_LABEL.get(zone_type)

    if label in DOCLAYNET_V1_LABELS:
        return label

    return None


def normalized_to_doclaynet_docling_document(
    *,
    normalized: dict[str, Any],
    name: str,
    page_width: float,
    page_height: float,
) -> dict[str, Any]:
    """
    Convert normalized Torvex output into asset-free DoclingDocument JSON
    suitable for DocLayNetV1 layout scoring.
    """
    texts: list[dict[str, Any]] = []
    pictures: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    body_children: list[dict[str, str]] = []

    table_seen_by_page: set[int] = set()
    formula_seen_by_page: set[int] = set()

    pages = normalized.get("pages") or []

    for raw_page in pages:
        page_no = int(raw_page.get("page_num", 0)) + 1

        for zone in raw_page.get("layout_zones") or []:
            zone_type = str(zone.get("type", ""))
            label = _doclaynet_label_for_zone(zone_type)

            if label is None:
                continue

            bbox = _bbox_from_pdfium(
                zone.get("bbox_pdfium"),
                page_width=page_width,
                page_height=page_height,
            )

            if bbox is None:
                continue

            if label == "table":
                table_index = len(tables)
                tables.append(
                    _make_table_item(
                        index=table_index,
                        page_no=page_no,
                        bbox=bbox,
                    )
                )
                body_children.append({"$ref": f"#/tables/{table_index}"})
                table_seen_by_page.add(page_no)
                continue

            if label == "picture":
                picture_index = len(pictures)
                pictures.append(
                    _make_picture_item(
                        index=picture_index,
                        page_no=page_no,
                        bbox=bbox,
                    )
                )
                body_children.append({"$ref": f"#/pictures/{picture_index}"})
                continue

            text = str(zone.get("zone_text") or zone.get("text") or label)

            text_index = len(texts)
            texts.append(
                _make_text_item(
                    index=text_index,
                    label=label,
                    text=text,
                    page_no=page_no,
                    bbox=bbox,
                )
            )
            body_children.append({"$ref": f"#/texts/{text_index}"})

            if label == "formula":
                formula_seen_by_page.add(page_no)

        # Fallback: if no table zone survived but table artifacts exist, export table bboxes.
        if page_no not in table_seen_by_page:
            for table in raw_page.get("tables") or []:
                bbox = _bbox_from_pdfium(
                    table.get("bbox_pdfium"),
                    page_width=page_width,
                    page_height=page_height,
                )

                if bbox is None:
                    continue

                table_index = len(tables)
                tables.append(
                    _make_table_item(
                        index=table_index,
                        page_no=page_no,
                        bbox=bbox,
                    )
                )
                body_children.append({"$ref": f"#/tables/{table_index}"})

        # Fallback: if formula_bboxes exist but formula zones did not survive.
        if page_no not in formula_seen_by_page:
            for formula_bbox in raw_page.get("formula_bboxes") or []:
                bbox = _bbox_from_pdfium(
                    formula_bbox,
                    page_width=page_width,
                    page_height=page_height,
                )

                if bbox is None:
                    continue

                text_index = len(texts)
                texts.append(
                    _make_text_item(
                        index=text_index,
                        label="formula",
                        text="formula",
                        page_no=page_no,
                        bbox=bbox,
                    )
                )
                body_children.append({"$ref": f"#/texts/{text_index}"})

    return {
        "schema_name": "DoclingDocument",
        "version": "1.10.0",
        "name": name,
        "furniture": {
            "self_ref": "#/furniture",
            "children": [],
            "content_layer": "furniture",
            "name": "_root_",
            "label": "unspecified",
        },
        "body": {
            "self_ref": "#/body",
            "children": body_children,
            "content_layer": "body",
            "name": "_root_",
            "label": "unspecified",
        },
        "groups": [],
        "texts": texts,
        "pictures": pictures,
        "tables": tables,
        "key_value_items": [],
        "form_items": [],
        "pages": {
            "1": {
                "size": {
                    "width": page_width,
                    "height": page_height,
                },
                "page_no": 1,
            }
        },
    }


def export_doclaynet_prediction(
    *,
    normalized: dict[str, Any],
    output_path: Path,
    name: str,
    page_width: float,
    page_height: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = normalized_to_doclaynet_docling_document(
        normalized=normalized,
        name=name,
        page_width=page_width,
        page_height=page_height,
    )

    output_path.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def generate_doclaynet_predictions(
    *,
    gt_dir: Path,
    prediction_dir: Path,
    limit: int,
    start_index: int = 0,
    overwrite: bool = False,
    save_normalized: bool = False,
    normalized_dir: Path | None = None,
    device: str = "cpu",
) -> DocLayNetPredictionSummary:
    requested = int(limit)
    processed = 0
    predictions_written = 0
    skipped_existing = 0
    errors = 0

    prediction_dir.mkdir(parents=True, exist_ok=True)

    if save_normalized:
        if normalized_dir is None:
            normalized_dir = prediction_dir.parent / "normalized"
        normalized_dir.mkdir(parents=True, exist_ok=True)

    adapter = TorvexExtractAdapter(device=device)

    for row in iter_gt_rows(gt_dir, limit=limit, start_index=start_index):
        doc_id = str(row["document_id"])
        output_path = prediction_dir / f"{doc_id}.json"

        if output_path.exists() and not overwrite:
            skipped_existing += 1
            continue

        try:
            page_width, page_height = _page_size_from_gt(row)

            with tempfile.TemporaryDirectory() as tmpdir:
                pdf_path = Path(tmpdir) / f"{doc_id}.pdf"
                pdf_path.write_bytes(row["BinaryDocument"])

                document = adapter.extract_document(pdf_path)
                normalized = normalize_document(document)

            if save_normalized and normalized_dir is not None:
                normalized_path = normalized_dir / f"{doc_id}.json"
                normalized_path.write_text(
                    json.dumps(normalized, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            export_doclaynet_prediction(
                normalized=normalized,
                output_path=output_path,
                name=doc_id,
                page_width=page_width,
                page_height=page_height,
            )

            processed += 1
            predictions_written += 1

        except Exception as exc:
            errors += 1
            print(f"[ERROR] {doc_id}: {exc}")

    return DocLayNetPredictionSummary(
        requested=requested,
        processed=processed,
        predictions_written=predictions_written,
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
        default=Path("benchmarks/docling_eval/DocLayNetV1_inspect/gt_dataset/test"),
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=Path("benchmarks/docling_eval/DocLayNetV1_inspect/predictions/torvex_extract"),
    )
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-normalized", action="store_true")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")

    args = parser.parse_args()

    summary = generate_doclaynet_predictions(
        gt_dir=args.gt_dir,
        prediction_dir=args.prediction_dir,
        limit=args.limit,
        start_index=args.start_index,
        overwrite=args.overwrite,
        save_normalized=args.save_normalized,
        device=args.device,
    )

    print("requested=", summary.requested)
    print("processed=", summary.processed)
    print("predictions_written=", summary.predictions_written)
    print("skipped_existing=", summary.skipped_existing)
    print("errors=", summary.errors)
    print("prediction_dir=", summary.prediction_dir)
    print("normalized_dir=", summary.normalized_dir)


if __name__ == "__main__":
    main()