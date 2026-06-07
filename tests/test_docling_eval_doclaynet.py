from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from torvex_bench.harnesses.docling_eval_doclaynet import (
    _bbox_from_pdfium,
    export_doclaynet_prediction,
    iter_gt_rows,
    normalized_to_doclaynet_docling_document,
)


def test_bbox_from_pdfium_exports_bottomleft_bbox() -> None:
    bbox = _bbox_from_pdfium(
        [10, 20, 100, 200],
        page_width=612,
        page_height=792,
    )

    assert bbox == {
        "l": 10.0,
        "t": 200.0,
        "r": 100.0,
        "b": 20.0,
        "coord_origin": "BOTTOMLEFT",
    }


def test_bbox_from_pdfium_rejects_invalid_bbox() -> None:
    bbox = _bbox_from_pdfium(
        [100, 20, 10, 200],
        page_width=612,
        page_height=792,
    )

    assert bbox is None


def test_normalized_to_doclaynet_docling_document_exports_layout_items() -> None:
    normalized = {
        "pages": [
            {
                "page_num": 0,
                "layout_zones": [
                    {
                        "type": "text",
                        "zone_text": "Body text",
                        "bbox_pdfium": [10, 20, 110, 120],
                    },
                    {
                        "type": "paragraph_title",
                        "zone_text": "Section",
                        "bbox_pdfium": [20, 130, 220, 160],
                    },
                    {
                        "type": "image",
                        "bbox_pdfium": [30, 170, 230, 270],
                    },
                    {
                        "type": "table",
                        "bbox_pdfium": [40, 280, 240, 380],
                    },
                    {
                        "type": "display_formula",
                        "bbox_pdfium": [50, 390, 250, 430],
                    },
                ],
                "tables": [],
                "formula_bboxes": [],
            }
        ]
    }

    doc = normalized_to_doclaynet_docling_document(
        normalized=normalized,
        name="sample-doc",
        page_width=612,
        page_height=792,
    )

    assert doc["schema_name"] == "DoclingDocument"
    assert doc["name"] == "sample-doc"
    assert doc["pages"]["1"]["size"] == {"width": 612, "height": 792}

    text_labels = [item["label"] for item in doc["texts"]]
    assert "text" in text_labels
    assert "section_header" in text_labels
    assert "formula" in text_labels

    assert len(doc["pictures"]) == 1
    assert doc["pictures"][0]["label"] == "picture"
    assert "image" not in doc["pictures"][0]

    assert len(doc["tables"]) == 1
    assert doc["tables"][0]["label"] == "table"

    assert "image" not in doc["pages"]["1"]


def test_normalized_to_doclaynet_docling_document_uses_table_fallback() -> None:
    normalized = {
        "pages": [
            {
                "page_num": 0,
                "layout_zones": [],
                "tables": [
                    {
                        "bbox_pdfium": [40, 280, 240, 380],
                    }
                ],
                "formula_bboxes": [],
            }
        ]
    }

    doc = normalized_to_doclaynet_docling_document(
        normalized=normalized,
        name="sample-doc",
        page_width=612,
        page_height=792,
    )

    assert len(doc["tables"]) == 1
    assert doc["tables"][0]["label"] == "table"


def test_normalized_to_doclaynet_docling_document_uses_formula_fallback() -> None:
    normalized = {
        "pages": [
            {
                "page_num": 0,
                "layout_zones": [],
                "tables": [],
                "formula_bboxes": [[50, 390, 250, 430]],
            }
        ]
    }

    doc = normalized_to_doclaynet_docling_document(
        normalized=normalized,
        name="sample-doc",
        page_width=612,
        page_height=792,
    )

    assert len(doc["texts"]) == 1
    assert doc["texts"][0]["label"] == "formula"


def test_export_doclaynet_prediction_writes_json(tmp_path: Path) -> None:
    output_path = tmp_path / "predictions" / "doc-id.json"

    export_doclaynet_prediction(
        normalized={
            "pages": [
                {
                    "page_num": 0,
                    "layout_zones": [
                        {
                            "type": "text",
                            "zone_text": "Hello",
                            "bbox_pdfium": [10, 20, 110, 120],
                        }
                    ],
                    "tables": [],
                    "formula_bboxes": [],
                }
            ]
        },
        output_path=output_path,
        name="doc-id",
        page_width=612,
        page_height=792,
    )

    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert data["name"] == "doc-id"
    assert data["texts"][0]["text"] == "Hello"


def test_iter_gt_rows_reads_parquet_rows(tmp_path: Path) -> None:
    gt_dir = tmp_path / "gt_dataset" / "test"
    gt_dir.mkdir(parents=True)

    table = pa.Table.from_pylist(
        [
            {"document_id": "doc-1", "BinaryDocument": b"%PDF-1"},
            {"document_id": "doc-2", "BinaryDocument": b"%PDF-2"},
        ]
    )
    pq.write_table(table, gt_dir / "shard.parquet")

    rows = list(iter_gt_rows(gt_dir, limit=1))

    assert len(rows) == 1
    assert rows[0]["document_id"] == "doc-1"