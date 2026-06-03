# this file is a translator

"""
Torvex raw output          Benchmark object
------------------------------------------------
raw_output["pdf"]      →   DocumentResult.pdf_path
raw_output["pages"]    →   DocumentResult.pages
raw_output["errors"]   →   DocumentResult.errors

page["page_num"]       →   PageResult.page_num
page["final_text"]     →   PageResult.text
page["tables"]         →   PageResult.tables
page["zones"]          →   PageResult.layout_zones
page["needs_ocr"]      →   PageResult.needs_ocr
page["ocr_reason"]     →   PageResult.metadata["ocr_reason"]
page["spotlight_bboxes"] → PageResult.spotlight_bboxes

table["rows"]          →   TableResult.rows
table["bbox_pdfium"]   →   TableResult.bbox_pdfium
table["bbox_plumber"]  →   TableResult.bbox_plumber
table["bbox_px"]       →   TableResult.bbox_px
table["source"]        →   TableResult.source
table["confidence"]    →   TableResult.confidence
table["table_id"]      →   TableResult.metadata["table_id"]
table["warnings"]      →   TableResult.metadata["warnings"]

"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from torvex_bench.adapters.base import ExtractionAdapter, DocumentResult, PageResult, TableResult


def get_formula_bboxes(page: dict[str, Any]) -> list[list[float]]:
    """
    Return formula bounding boxes for one page.

    Priority:
    1. Use page["formula_bboxes"] if Phase 1 already provides it.
    2. Otherwise derive formula boxes from page["zones"].
    """
    direct_bboxes = page.get("formula_bboxes") or []

    if direct_bboxes:
        return [
            [float(value) for value in bbox]
            for bbox in direct_bboxes
            if bbox is not None
        ]

    formula_bboxes: list[list[float]] = []

    for zone in page.get("zones") or []:
        zone_type = str(zone.get("type", ""))

        if "formula" not in zone_type:
            continue

        bbox = zone.get("bbox_pdfium")

        if bbox is None:
            continue

        formula_bboxes.append([float(value) for value in bbox])

    return formula_bboxes


def _bbox_to_float_list(bbox: Any) -> list[float] | None:
    """
    Convert a raw bbox into a list of floats.

    Missing bbox stays None.
    """
    if bbox is None:
        return None

    return [float(value) for value in bbox]


def _bboxes_to_float_lists(bboxes: Any) -> list[list[float]]:
    """
    Convert many raw bboxes into lists of floats.
    """
    if not bboxes:
        return []

    clean_bboxes: list[list[float]] = []

    for bbox in bboxes:
        clean_bbox = _bbox_to_float_list(bbox)

        if clean_bbox is not None:
            clean_bboxes.append(clean_bbox)

    return clean_bboxes


def convert_table(raw_table: dict[str, Any]) -> TableResult:
    """
    Convert one raw Torvex table dictionary into TableResult.
    """
    rows: list[list[str]] = []

    for row in raw_table.get("rows") or []:
        clean_row = ["" if cell is None else str(cell) for cell in row]
        rows.append(clean_row)

    raw_confidence = raw_table.get("confidence")

    metadata: dict[str, Any] = {}

    for key in ("table_id", "kind", "method", "warnings"):
        if key in raw_table:
            metadata[key] = raw_table[key]

    return TableResult(
        rows=rows,
        bbox_pdfium=_bbox_to_float_list(raw_table.get("bbox_pdfium")),
        bbox_plumber=_bbox_to_float_list(raw_table.get("bbox_plumber")),
        bbox_px=_bbox_to_float_list(raw_table.get("bbox_px")),
        source=str(raw_table.get("source") or "unknown"),
        confidence=1.0 if raw_confidence is None else float(raw_confidence),
        metadata=metadata,
    )


def convert_page(raw_page: dict[str, Any]) -> PageResult:
    """
    Convert one raw Torvex page dictionary into PageResult.
    """
    needs_ocr = bool(raw_page.get("needs_ocr", False))

    metadata: dict[str, Any] = dict(raw_page.get("metadata") or {})

    for key in (
        "is_tagged",
        "ocr_reason",
        "page_width",
        "page_height",
        "effective_page_width_pt",
        "effective_page_height_pt",
        "has_bordered_table",
        "layout_grade",
        "page_class",
    ):
        if key in raw_page:
            metadata[key] = raw_page[key]

    return PageResult(
        page_num=int(raw_page.get("page_num", 0)),
        text=str(raw_page.get("final_text") or raw_page.get("text") or ""),
        tables=[
            convert_table(raw_table)
            for raw_table in raw_page.get("tables") or []
        ],
        layout_zones=list(raw_page.get("zones") or []),
        formula_bboxes=get_formula_bboxes(raw_page),
        spotlight_bboxes=_bboxes_to_float_lists(
            raw_page.get("spotlight_bboxes")
        ),
        needs_ocr=needs_ocr,
        ocr_used=bool(raw_page.get("ocr_used", needs_ocr)),
        metadata=metadata,
    )


def convert_document(
    raw_output: dict[str, Any],
    pdf_path: str | None = None,
) -> DocumentResult:
    """
    Convert one full raw Torvex output dictionary into DocumentResult.
    """
    metadata: dict[str, Any] = dict(raw_output.get("metadata") or {})

    return DocumentResult(
        pdf_path=str(pdf_path or raw_output.get("pdf") or ""),
        pages=[
            convert_page(raw_page)
            for raw_page in raw_output.get("pages") or []
        ],
        errors=list(raw_output.get("errors") or []),
        metadata=metadata,
    )


class TorvexExtractAdapter(ExtractionAdapter):
    """
    Real Torvex Extract adapter.

    Calls torvex-extract engine, then converts raw output
    into benchmark DocumentResult.
    """

    name = "torvex_extract"
    version = "0.1.0"

    def __init__(self) -> None:
        self._warmed = False

    def _ensure_warmed(self) -> None:
        if self._warmed:
            return

        from torvex_extract.visual_zoning import engine

        if not engine.is_warmed():
            engine.warm()

        self._warmed = True

    def extract_document(self, pdf_path: str | Path) -> DocumentResult:
        self._ensure_warmed()

        from torvex_extract import extract_with_pypdfium2

        pages, errors = extract_with_pypdfium2(str(pdf_path))

        raw_output = {
            "pdf": str(pdf_path),
            "pages": pages,
            "errors": errors,
            "metadata": {
                "adapter": self.name,
                "adapter_version": self.version,
            },
        }

        return convert_document(raw_output, pdf_path=str(pdf_path))

    # FIX: extract() alias removed from here.
    # It now lives on ExtractionAdapter (base.py) as a concrete method
    # so all three adapters expose it symmetrically without duplication.