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


def _formula_bbox(formula: dict[str, Any]) -> list[float] | None:
    """
    Prefer bbox_px because layout formula zones also use bbox_px in scanned OmniDocBench.
    """
    return (
        _bbox(formula.get("bbox_px"))
        or _bbox(formula.get("bbox_plumber"))
        or _bbox(formula.get("bbox_pdfium"))
    )


def _best_formula_for_zone(
    zone: dict[str, Any],
    formulas: list[dict[str, Any]],
    used_formula_indexes: set[int],
) -> int | None:
    """
    Match one layout display_formula zone to one normalized formula artifact.

    This prevents accepted display formulas from being appended at page end,
    which hurts OmniDocBench reading_order.
    """
    zone_box = _zone_bbox(zone)

    best_index: int | None = None
    best_score = 0.0

    for index, formula in enumerate(formulas):
        if index in used_formula_indexes:
            continue

        if not formula_to_markdown(formula):
            continue

        score = _bbox_iou(zone_box, _formula_bbox(formula))
        if score > best_score:
            best_score = score
            best_index = index

    if best_index is None:
        return None

    if best_score < 0.05:
        return None

    return best_index


def _formulas_for_zone(
    zone: dict[str, Any],
    formulas: list[dict[str, Any]],
    used_formula_indexes: set[int],
) -> list[int]:
    """
    2026-06-11 display-formula splitter support.

    torvex-extract now splits merged PP-DocLayoutV3 display boxes, so ONE
    layout display_formula zone legitimately maps to MANY formula artifacts.
    The old _best_formula_for_zone emitted only the single best-IoU formula;
    the leftover split children fell through to the append-at-page-end loop,
    which re-broke reading order. This returns ALL emittable formulas that
    sit inside the zone, sorted top-to-bottom.

    Overlap is intersection-over-FORMULA-area (>= 0.5), not IoU: split
    children are small relative to the parent zone, so IoU is misleadingly
    low even for a perfect child. Plain IoU >= 0.05 is kept as a fallback
    for the unsplit one-formula case.
    """
    zone_box = _zone_bbox(zone)
    if zone_box is None:
        return []

    matched: list[tuple[float, int]] = []

    for index, formula in enumerate(formulas):
        if index in used_formula_indexes:
            continue
        if not formula_to_markdown(formula):
            continue

        formula_box = _formula_bbox(formula)
        if formula_box is None:
            continue

        ix0 = max(zone_box[0], formula_box[0])
        iy0 = max(zone_box[1], formula_box[1])
        ix1 = min(zone_box[2], formula_box[2])
        iy1 = min(zone_box[3], formula_box[3])

        if ix1 <= ix0 or iy1 <= iy0:
            continue

        inter = (ix1 - ix0) * (iy1 - iy0)
        formula_area = (formula_box[2] - formula_box[0]) * (formula_box[3] - formula_box[1])

        inside_enough = formula_area > 0 and (inter / formula_area) >= 0.5
        iou_enough = _bbox_iou(zone_box, formula_box) >= 0.05

        if inside_enough or iou_enough:
            matched.append((formula_box[1], index))

    matched.sort()
    return [index for _, index in matched]


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
    Interleave single-table pages safely.

    2026-06-11:
    Interleave accepted display formulas at their layout-zone position.

    Why:
        Formula-enabled runs improved formula edit quality, but reading_order dropped
        because formulas were appended after all page text. This patch emits display
        formulas where their display_formula layout zones appear.
    """
    tables = list(page.get("tables") or [])
    formulas = list(page.get("formulas") or [])
    zones = list(page.get("layout_zones") or [])

    table_zone_count = sum(1 for zone in zones if str(zone.get("type") or "") == "table")
    formula_zone_count = sum(
        1 for zone in zones if str(zone.get("type") or "") == "display_formula"
    )

    should_interleave_single_table = bool(
        tables and zones and len(tables) == 1 and table_zone_count == 1
    )
    should_interleave_formulas = bool(
        formulas and zones and formula_zone_count > 0
    )

    if not should_interleave_single_table and not should_interleave_formulas:
        return _fallback_page_to_markdown(page)

    blocks: list[str] = []
    used_table_indexes: set[int] = set()
    used_formula_indexes: set[int] = set()

    skip_zone_types = {
        "image",
        "chart",
        "figure",
        "header_image",
        "footer_image",
        "seal",
        "inline_formula",
        "formula_number",
    }

    for zone in zones:
        zone_type = str(zone.get("type") or "")

        if zone_type == "table":
            if should_interleave_single_table:
                table_index = _best_table_for_zone(zone, tables, used_table_indexes)
                if table_index is not None:
                    table_html = table_rows_to_html(tables[table_index].get("rows") or [])
                    if table_html:
                        blocks.append(table_html)
                        used_table_indexes.add(table_index)
            continue

        if zone_type == "display_formula":
            if should_interleave_formulas:
                # 2026-06-11 display-formula splitter: one zone can now map
                # to MANY formulas. Emit every match here, top-to-bottom,
                # instead of one best-IoU formula (which pushed split
                # siblings to the page-end leftover loop and broke
                # reading_order).
                for formula_index in _formulas_for_zone(
                    zone,
                    formulas,
                    used_formula_indexes,
                ):
                    formula_md = formula_to_markdown(formulas[formula_index])
                    if formula_md:
                        blocks.append(formula_md)
                        used_formula_indexes.add(formula_index)
            continue

        if zone_type in skip_zone_types:
            continue

        zone_text = _clean_text(zone.get("zone_text") or zone.get("text"))
        if zone_text:
            blocks.append(zone_text)

    for index, formula in enumerate(formulas):
        if index in used_formula_indexes:
            continue

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