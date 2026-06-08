"""
olmOCR-Bench Markdown exporter.

Purpose
-------
Translate torvex-bench normalized output into the Markdown prediction files
expected by the official olmOCR-Bench evaluator.

Official prediction filename pattern
------------------------------------
The official evaluator expects files like:

    <pdf_stem>_pg<page>_repeat<repeat>.md

Example:

    old_scans/1_pg1_repeat1.md

This exporter only writes Markdown files.
It does not:
    - prepare the dataset
    - run extraction
    - call official evaluator
    - compute metrics

Tables
------
olmOCR-Bench table tests parse both Markdown and HTML tables.
HTML is safer because cell contents may contain pipes/newlines/markdown syntax.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any


def _clean_text(value: Any) -> str:
    """
    Convert any value into safe plain text for Markdown/HTML output.
    """
    if value is None:
        return ""

    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def table_rows_to_html(rows: list[list[Any]]) -> str:
    """
    Convert rows[][] into a simple HTML table.

    Input:
        [["A", "B"], ["1", "2"]]

    Output:
        <table>
        <tr><td>A</td><td>B</td></tr>
        <tr><td>1</td><td>2</td></tr>
        </table>
    """
    if not rows:
        return ""

    html_lines = ["<table>"]

    for row in rows:
        cells: list[str] = []

        for cell in row or []:
            clean_cell = _clean_text(cell).replace("\n", " ")
            cells.append(f"<td>{escape(clean_cell)}</td>")

        html_lines.append(f"<tr>{''.join(cells)}</tr>")

    html_lines.append("</table>")
    return "\n".join(html_lines)


def normalized_page_to_markdown(page: dict[str, Any]) -> str:
    """
    Convert one normalized page dictionary into olmOCR-Bench Markdown.

    Current safe order:
        1. Page text
        2. Tables as HTML

    Later we can improve this by interleaving text/table blocks by bbox order.
    """
    blocks: list[str] = []

    text = _clean_text(page.get("text"))
    if text:
        blocks.append(text)

    for table in page.get("tables") or []:
        rows = table.get("rows") or []
        table_html = table_rows_to_html(rows)
        if table_html:
            blocks.append(table_html)

    markdown = "\n\n".join(blocks).strip()

    if markdown:
        return markdown + "\n"

    return ""


def _page_matches_official_page(page: dict[str, Any], official_page: int) -> bool:
    """
    Match an official 1-based olmOCR page number against normalized page_num.

    Torvex normalized outputs have historically used page_num either as:
        - 0-based index
        - 1-based page number

    For olmOCR-Bench PDFs, most files are single-page PDFs and tests use page=1.
    This helper avoids failing because of indexing convention.
    """
    page_num = int(page.get("page_num", 0))

    return page_num == official_page or page_num == official_page - 1


def normalized_document_page_to_markdown(
    document: dict[str, Any],
    *,
    page: int = 1,
) -> str:
    """
    Convert one page from a normalized document into Markdown.

    If the requested page is not found but the document has exactly one page,
    use that page. This keeps single-page benchmark PDFs robust.
    """
    pages = list(document.get("pages") or [])

    for normalized_page in pages:
        if _page_matches_official_page(normalized_page, page):
            return normalized_page_to_markdown(normalized_page)

    if len(pages) == 1:
        return normalized_page_to_markdown(pages[0])

    return ""


def export_olmocr_markdown_prediction(
    normalized_document: dict[str, Any],
    prediction_path: str | Path,
    *,
    page: int = 1,
) -> Path:
    """
    Write one olmOCR-Bench .md prediction.

    Args:
        normalized_document:
            Output from normalize_document(...).

        prediction_path:
            Full official prediction path, usually:
                <bench_data>/torvex_extract/<pdf_stem>_pg1_repeat1.md

        page:
            Official 1-based page number from JSONL tests.

    Returns:
        Written prediction path.
    """
    prediction_path = Path(prediction_path)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)

    markdown = normalized_document_page_to_markdown(
        normalized_document,
        page=page,
    )

    prediction_path.write_text(markdown, encoding="utf-8")
    return prediction_path