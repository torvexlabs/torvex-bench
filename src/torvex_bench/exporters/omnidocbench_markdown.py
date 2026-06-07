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


def normalized_page_to_markdown(page: dict[str, Any]) -> str:
    """
    Convert one normalized page dictionary into OmniDocBench Markdown.

    Current order:
        1. Page text as plain paragraphs.
        2. Tables as HTML tables.

    This is the safe first implementation. Later, if needed, we can improve
    reading order by interleaving text/table blocks using bbox order.
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

    return "\n\n".join(blocks).strip() + "\n"


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