from typing import Any

from torvex_bench.adapters.base import DocumentResult, PageResult, TableResult


def normalize_table(table: TableResult) -> dict[str, Any]:
    """
    Convert one TableResult into a JSON-safe benchmark dictionary.
    """
    rows: list[list[str]] = []

    for row in table.rows or []:
        clean_row = ["" if cell is None else str(cell) for cell in row]
        rows.append(clean_row)

    return {
        "rows": rows,
        "bbox_pdfium": (
            [float(value) for value in table.bbox_pdfium]
            if table.bbox_pdfium is not None
            else None
        ),
        "bbox_plumber": (
            [float(value) for value in table.bbox_plumber]
            if table.bbox_plumber is not None
            else None
        ),
        "bbox_px": (
            [float(value) for value in table.bbox_px]
            if table.bbox_px is not None
            else None
        ),
        "source": str(table.source),
        "confidence": float(table.confidence),
        "metadata": dict(table.metadata),
    }


def normalize_page(page: PageResult) -> dict[str, Any]:
    """
    Convert one PageResult into a JSON-safe benchmark dictionary.
    """
    return {
        "page_num": int(page.page_num),
        "text": str(page.text or ""),
        "tables": [normalize_table(table) for table in page.tables],
        "layout_zones": list(page.layout_zones),
        "formula_bboxes": [
            [float(value) for value in bbox]
            for bbox in page.formula_bboxes
        ],
        "formulas": [dict(formula) for formula in page.formulas],
        "spotlight_bboxes": [
            [float(value) for value in bbox]
            for bbox in page.spotlight_bboxes
        ],
        "needs_ocr": bool(page.needs_ocr),
        "ocr_used": bool(page.ocr_used),
        "metadata": dict(page.metadata),
    }


def normalize_document(document: DocumentResult) -> dict[str, Any]:
    """
    Convert one DocumentResult into a JSON-safe benchmark dictionary.
    """
    return {
        "pdf_path": str(document.pdf_path),
        "pages": [normalize_page(page) for page in document.pages],
        "errors": list(document.errors),
        "metadata": dict(document.metadata),
    }