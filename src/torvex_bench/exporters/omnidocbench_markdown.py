"""
OmniDocBench markdown exporter.

Purpose
-------
Translate torvex-bench normalized output into the prediction format expected by
the official OmniDocBench end-to-end evaluator.

Input:
    normalize_document(DocumentResult) dictionary

Output:
    one Markdown file per OmniDocBench page image:

        <image_stem>.md

Official OmniDocBench behavior we inspected
-------------------------------------------
The official evaluator reads a UTF-8 Markdown file for each page. The filename
must match page_info.image_path with the image suffix replaced by .md.

Example:
    GT image_path:
        page-d1561665-5359-42fe-920c-d6e3bff81953.png

    Prediction:
        page-d1561665-5359-42fe-920c-d6e3bff81953.md

Tables
------
The official parser recognizes HTML tables directly. Therefore this exporter
writes tables as:

    <table>
      <tr><td>...</td></tr>
    </table>

This is safer than pipe-style Markdown tables because pipe tables can be brittle
when cells contain pipes, newlines, or markdown syntax.

Scope
-----
No metric logic.
No scoring.
No GT parsing.
No official harness call.

Formula CDM is intentionally excluded in the official config, so this exporter
does not attempt LaTeX/formula reconstruction.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any


def _clean_text(value: Any) -> str:
    """
    Convert any value into safe plain text for Markdown/HTML output.

    None becomes empty string.
    Newlines inside table cells become spaces.
    """
    if value is None:
        return ""

    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def _clean_latex_block(value: Any) -> str:
    """
    Return clean LaTeX content without outer display delimiters.
    """
    latex = _clean_text(value)

    if latex.startswith("$$") and latex.endswith("$$") and len(latex) >= 4:
        latex = latex[2:-2].strip()

    if latex.startswith("\\[") and latex.endswith("\\]"):
        latex = latex[2:-2].strip()

    return latex


def table_rows_to_html(rows: list[list[Any]]) -> str:
    """
    Convert normalized table rows into a simple HTML table.

    Input:
        [["A", "B"], ["1", "2"]]

    Output:
        <table>
        <tr><td>A</td><td>B</td></tr>
        <tr><td>1</td><td>2</td></tr>
        </table>

    Empty tables return an empty string.
    """
    if not rows:
        return ""

    html_lines = ["<table>"]

    for row in rows:
        cells = []
        for cell in row or []:
            clean_cell = _clean_text(cell).replace("\n", " ")
            cells.append(f"<td>{escape(clean_cell)}</td>")

        html_lines.append(f"<tr>{''.join(cells)}</tr>")

    html_lines.append("</table>")
    return "\n".join(html_lines)


def formula_to_markdown(formula: dict[str, Any]) -> str:
    """
    Convert one normalized formula artifact into display math Markdown.

    Only emits accepted / low_confidence display formulas.
    Inline reinsertion is intentionally deferred.
    """
    formula_type = str(formula.get("type") or "")
    status = str(formula.get("status") or "")
    latex = _clean_latex_block(formula.get("latex"))

    if formula_type != "display_formula":
        return ""

    if status not in {"accepted", "low_confidence"}:
        return ""

    if not latex:
        return ""

    return f"$$\n{latex}\n$$"


def _bbox(value: Any) -> list[float] | None:
    """
    Normalize bbox into [x0, y0, x1, y1].

    2026-06-10:
    Added for OmniDocBench table-order tuning.
    The benchmark exporter can interleave structured table HTML at the
    table zone position when the engine already provides bbox/layout zones.
    """
    if value is None:
        return None

    try:
        box = [float(v) for v in value]
    except Exception:
        return None

    if len(box) != 4:
        return None

    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return None

    return [x0, y0, x1, y1]


def _bbox_iou(a: list[float] | None, b: list[float] | None) -> float:
    """
    Return IoU for two same-coordinate bboxes.

    2026-06-10:
    Used only for matching a layout table zone to the structured table artifact.
    """
    if a is None or b is None:
        return 0.0

    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)

    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0

    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)

    union = area_a + area_b - inter
    if union <= 0:
        return 0.0

    return inter / union


def _table_bbox(table: dict[str, Any]) -> list[float] | None:
    """
    Prefer bbox_px because layout_zones also use bbox_px in scanned OmniDocBench.

    2026-06-10:
    Keep fallback bboxes only for robustness, but normal run_020 artifacts
    contain bbox_px for both table zones and table artifacts.
    """
    return (
        _bbox(table.get("bbox_px"))
        or _bbox(table.get("bbox_plumber"))
        or _bbox(table.get("bbox_pdfium"))
    )


def _zone_bbox(zone: dict[str, Any]) -> list[float] | None:
    return (
        _bbox(zone.get("bbox_px"))
        or _bbox(zone.get("bbox_plumber"))
        or _bbox(zone.get("bbox_pdfium"))
    )


def _best_table_for_zone(
    zone: dict[str, Any],
    tables: list[dict[str, Any]],
    used_table_indexes: set[int],
) -> int | None:
    """
    Match one layout table zone to one structured table artifact.

    2026-06-10:
    Fixes OmniDocBench exporter bug where tables were always appended at page end.
    """
    zone_box = _zone_bbox(zone)

    best_index: int | None = None
    best_score = 0.0

    for index, table in enumerate(tables):
        if index in used_table_indexes:
            continue

        score = _bbox_iou(zone_box, _table_bbox(table))
        if score > best_score:
            best_score = score
            best_index = index

    if best_index is None:
        return None

    # Small threshold because some table boxes are not identical but overlap well enough.
    if best_score < 0.05:
        return None

    return best_index


def _fallback_page_to_markdown(page: dict[str, Any]) -> str:
    """
    Original safe exporter behavior.

    2026-06-10:
    Kept as fallback so pure text pages and old tests remain unchanged.
    """
    blocks: list[str] = []

    text = _clean_text(page.get("text"))
    if text:
        blocks.append(text)

    for formula in page.get("formulas") or []:
        formula_md = formula_to_markdown(formula)
        if formula_md:
            blocks.append(formula_md)

    for table in page.get("tables") or []:
        rows = table.get("rows") or []
        table_html = table_rows_to_html(rows)
        if table_html:
            blocks.append(table_html)

    return "\n\n".join(blocks).strip() + "\n"


def normalized_page_to_markdown(page: dict[str, Any]) -> str:
    """
    Convert one normalized page dictionary into OmniDocBench Markdown.

    2026-06-10:
    Narrow table-order patch.

    Why:
        run_020 inspection showed pages where the engine had a table layout zone
        before body text, but this exporter appended the HTML table after all text.
        That hurts OmniDocBench reading_order and table-page matching.

    Safety:
        - Only activates when both tables and table layout zones exist.
        - Pure text pages still use the old page["text"] fallback.
        - If table/zone matching fails, unmatched tables are appended as before.
    """
    tables = list(page.get("tables") or [])
    zones = list(page.get("layout_zones") or [])

    table_zone_count = sum(1 for zone in zones if str(zone.get("type") or "") == "table")

    # 2026-06-10:
    # Keep this patch intentionally narrow.
    # Single-table pages improved in run_021, but multi-table double-column pages
    # can regress when the engine zone order places the right table before the left.
    # Until engine multi-column table ordering is fixed, only interleave exactly one
    # structured table with exactly one table zone.
    if not tables or not zones or len(tables) != 1 or table_zone_count != 1:
        return _fallback_page_to_markdown(page)

    blocks: list[str] = []
    used_table_indexes: set[int] = set()

    skip_zone_types = {
        "image",
        "chart",
        "figure",
        "header_image",
        "footer_image",
        "seal",
        "inline_formula",
        "display_formula",
        "formula_number",
    }

    for zone in zones:
        zone_type = str(zone.get("type") or "")

        if zone_type == "table":
            table_index = _best_table_for_zone(zone, tables, used_table_indexes)
            if table_index is not None:
                table_html = table_rows_to_html(tables[table_index].get("rows") or [])
                if table_html:
                    blocks.append(table_html)
                    used_table_indexes.add(table_index)
            continue

        if zone_type in skip_zone_types:
            continue

        zone_text = _clean_text(zone.get("zone_text") or zone.get("text"))
        if zone_text:
            blocks.append(zone_text)

    for formula in page.get("formulas") or []:
        formula_md = formula_to_markdown(formula)
        if formula_md:
            blocks.append(formula_md)

    for index, table in enumerate(tables):
        if index in used_table_indexes:
            continue

        table_html = table_rows_to_html(table.get("rows") or [])
        if table_html:
            blocks.append(table_html)

    markdown = "\n\n".join(block for block in blocks if block).strip()
    if markdown:
        return markdown + "\n"

    return _fallback_page_to_markdown(page)

def normalized_document_to_markdown(document: dict[str, Any]) -> str:
    """
    Convert a normalized DocumentResult dictionary into one Markdown page.

    OmniDocBench samples are one page each. The temporary PDF created from the
    page image should normally return exactly one page. If an adapter returns
    more than one page, we concatenate them in page order instead of failing.
    """
    pages = list(document.get("pages") or [])
    pages.sort(key=lambda page: int(page.get("page_num", 0)))

    markdown_pages = [
        normalized_page_to_markdown(page).strip()
        for page in pages
    ]

    markdown = "\n\n".join(block for block in markdown_pages if block)

    if markdown:
        return markdown.strip() + "\n"

    return ""


def export_omnidocbench_markdown_prediction(
    normalized_document: dict[str, Any],
    prediction_path: str | Path,
) -> Path:
    """
    Write one OmniDocBench .md prediction file.

    Args:
        normalized_document:
            Output from normalize_document(...).

        prediction_path:
            Full path to the official prediction file, usually:
            benchmarks/omnidocbench/OmniDocBench_scanned/predictions/torvex_extract/<image_stem>.md

    Returns:
        Path to the written .md file.
    """
    prediction_path = Path(prediction_path)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)

    markdown = normalized_document_to_markdown(normalized_document)
    prediction_path.write_text(markdown, encoding="utf-8")

    return prediction_path


def export_sample_markdown_prediction(
    normalized_document: dict[str, Any],
    *,
    prediction_filename: str,
    predictions_dir: str | Path,
) -> Path:
    """
    Write prediction for one OmniDocBench sample.

    prediction_filename comes from OmniDocBenchSample.prediction_filename.

    Example:
        prediction_filename = "page-d1561665-5359-42fe-920c-d6e3bff81953.md"
    """
    return export_omnidocbench_markdown_prediction(
        normalized_document,
        Path(predictions_dir) / prediction_filename,
    )