from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _cell_bbox(row: int, col: int, cell_w: float = 100.0, cell_h: float = 24.0) -> dict[str, Any]:
    return {
        "l": col * cell_w,
        "t": row * cell_h,
        "r": (col + 1) * cell_w,
        "b": (row + 1) * cell_h,
        "coord_origin": "TOPLEFT",
    }


def rows_to_docling_document(
    *,
    rows: list[list[str]],
    name: str,
    page_width: float | None = None,
    page_height: float | None = None,
) -> dict[str, Any]:
    num_rows = len(rows)
    num_cols = max((len(row) for row in rows), default=0)

    width = float(page_width or max(1, num_cols) * 100.0)
    height = float(page_height or max(1, num_rows) * 24.0)

    table_cells: list[dict[str, Any]] = []

    for row_idx, row in enumerate(rows):
        padded_row = list(row) + [""] * (num_cols - len(row))

        for col_idx, text in enumerate(padded_row):
            table_cells.append(
                {
                    "bbox": _cell_bbox(row_idx, col_idx),
                    "row_span": 1,
                    "col_span": 1,
                    "start_row_offset_idx": row_idx,
                    "end_row_offset_idx": row_idx + 1,
                    "start_col_offset_idx": col_idx,
                    "end_col_offset_idx": col_idx + 1,
                    "text": "" if text is None else str(text),
                    "column_header": False,
                    "row_header": False,
                    "row_section": False,
                    "fillable": False,
                }
            )

    grid = [
        [{"$ref": f"#/tables/0/data/table_cells/{row_idx * num_cols + col_idx}"} for col_idx in range(num_cols)]
        for row_idx in range(num_rows)
    ]

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
            "children": [{"$ref": "#/tables/0"}] if num_rows and num_cols else [],
            "content_layer": "body",
            "name": "_root_",
            "label": "unspecified",
        },
        "groups": [],
        "texts": [],
        "pictures": [],
        "tables": [
            {
                "self_ref": "#/tables/0",
                "parent": {"$ref": "#/body"},
                "children": [],
                "content_layer": "body",
                "label": "table",
                "prov": [
                    {
                        "page_no": 1,
                        "bbox": {
                            "l": 0.0,
                            "t": height,
                            "r": width,
                            "b": 0.0,
                            "coord_origin": "BOTTOMLEFT",
                        },
                        "charspan": [0, 0],
                    }
                ],
                "captions": [],
                "references": [],
                "footnotes": [],
                "data": {
                    "table_cells": table_cells,
                    "num_rows": num_rows,
                    "num_cols": num_cols,
                    "orientation": "rot_0",
                    "grid": grid,
                },
                "annotations": [],
            }
        ] if num_rows and num_cols else [],
        "key_value_items": [],
        "form_items": [],
        "pages": {
            "1": {
                "size": {
                    "width": width,
                    "height": height,
                },
                "page_no": 1,
            }
        },
    }


def export_rows_as_docling_json(
    *,
    rows: list[list[str]],
    output_path: Path,
    name: str,
    page_width: float | None = None,
    page_height: float | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = rows_to_docling_document(
        rows=rows,
        name=name,
        page_width=page_width,
        page_height=page_height,
    )

    output_path.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
