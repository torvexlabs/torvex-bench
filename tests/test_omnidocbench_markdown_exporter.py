from pathlib import Path

from torvex_bench.exporters.omnidocbench_markdown import (
    export_omnidocbench_markdown_prediction,
    export_sample_markdown_prediction,
    normalized_document_to_markdown,
    normalized_page_to_markdown,
    table_rows_to_html,
)


def test_table_rows_to_html_escapes_cells() -> None:
    html = table_rows_to_html(
        [
            ["A&B", "<tag>"],
            ["1", "2"],
        ]
    )

    assert html == (
        "<table>\n"
        "<tr><td>A&amp;B</td><td>&lt;tag&gt;</td></tr>\n"
        "<tr><td>1</td><td>2</td></tr>\n"
        "</table>"
    )


def test_table_rows_to_html_returns_empty_for_empty_rows() -> None:
    assert table_rows_to_html([]) == ""


def test_normalized_page_to_markdown_writes_text_then_html_table() -> None:
    page = {
        "page_num": 0,
        "text": "Hello page",
        "tables": [
            {
                "rows": [
                    ["Col A", "Col B"],
                    ["1", "2"],
                ]
            }
        ],
    }

    markdown = normalized_page_to_markdown(page)

    assert markdown == (
        "Hello page\n\n"
        "<table>\n"
        "<tr><td>Col A</td><td>Col B</td></tr>\n"
        "<tr><td>1</td><td>2</td></tr>\n"
        "</table>\n"
    )


def test_normalized_document_to_markdown_sorts_pages() -> None:
    document = {
        "pdf_path": "sample.pdf",
        "pages": [
            {
                "page_num": 2,
                "text": "Second",
                "tables": [],
            },
            {
                "page_num": 1,
                "text": "First",
                "tables": [],
            },
        ],
        "errors": [],
        "metadata": {},
    }

    markdown = normalized_document_to_markdown(document)

    assert markdown == "First\n\nSecond\n"


def test_export_omnidocbench_markdown_prediction_writes_file(tmp_path: Path) -> None:
    document = {
        "pdf_path": "sample.pdf",
        "pages": [
            {
                "page_num": 0,
                "text": "Hello",
                "tables": [],
            }
        ],
        "errors": [],
        "metadata": {},
    }

    prediction_path = tmp_path / "page-abc.md"
    result_path = export_omnidocbench_markdown_prediction(
        document,
        prediction_path,
    )

    assert result_path == prediction_path
    assert prediction_path.read_text(encoding="utf-8") == "Hello\n"


def test_export_sample_markdown_prediction_uses_prediction_filename(
    tmp_path: Path,
) -> None:
    document = {
        "pdf_path": "sample.pdf",
        "pages": [
            {
                "page_num": 0,
                "text": "Sample text",
                "tables": [],
            }
        ],
        "errors": [],
        "metadata": {},
    }

    result_path = export_sample_markdown_prediction(
        document,
        prediction_filename="page-abc.md",
        predictions_dir=tmp_path,
    )

    assert result_path == tmp_path / "page-abc.md"
    assert result_path.read_text(encoding="utf-8") == "Sample text\n"


def test_omnidocbench_markdown_emits_display_formula_latex() -> None:
    from torvex_bench.exporters.omnidocbench_markdown import normalized_page_to_markdown

    page = {
        "text": "Before formula.",
        "formulas": [
            {
                "type": "display_formula",
                "latex": r"\frac{a}{b}",
                "status": "accepted",
            }
        ],
        "tables": [],
    }

    markdown = normalized_page_to_markdown(page)

    assert "Before formula." in markdown
    assert "$$\n\\frac{a}{b}\n$$" in markdown

def test_normalized_page_to_markdown_interleaves_table_by_layout_zone() -> None:
    """
    2026-06-10:
    Regression test for OmniDocBench table reading-order bug.

    The old exporter wrote page text first and appended tables at the end.
    When layout_zones contain a table zone, the structured HTML table should
    appear at that zone position.
    """
    page = {
        "page_num": 0,
        "text": "Before\nAfter",
        "layout_zones": [
            {
                "type": "text",
                "bbox_px": [0, 0, 100, 20],
                "zone_text": "Before",
            },
            {
                "type": "table",
                "bbox_px": [0, 30, 100, 80],
            },
            {
                "type": "text",
                "bbox_px": [0, 90, 100, 120],
                "zone_text": "After",
            },
        ],
        "tables": [
            {
                "bbox_px": [0, 30, 100, 80],
                "rows": [
                    ["A", "B"],
                    ["1", "2"],
                ],
            }
        ],
    }

    markdown = normalized_page_to_markdown(page)

    assert markdown == (
        "Before\n\n"
        "<table>\n"
        "<tr><td>A</td><td>B</td></tr>\n"
        "<tr><td>1</td><td>2</td></tr>\n"
        "</table>\n\n"
        "After\n"
    )

def test_normalized_page_to_markdown_keeps_fallback_for_multi_table_pages() -> None:
    """
    2026-06-10:
    Regression guard after run_021.

    Single-table interleaving improved page-573, but multi-table page-835
    regressed because engine zone order placed the right table before the left.
    Multi-table pages should stay on the old fallback path until engine/table
    ordering is fixed.
    """
    page = {
        "page_num": 0,
        "text": "Body text",
        "layout_zones": [
            {"type": "table", "bbox_px": [100, 0, 200, 100]},
            {"type": "table", "bbox_px": [0, 0, 90, 100]},
            {"type": "text", "bbox_px": [0, 110, 200, 140], "zone_text": "Body text"},
        ],
        "tables": [
            {"bbox_px": [100, 0, 200, 100], "rows": [["R"]]},
            {"bbox_px": [0, 0, 90, 100], "rows": [["L"]]},
        ],
    }

    markdown = normalized_page_to_markdown(page)

    assert markdown == (
        "Body text\n\n"
        "<table>\n<tr><td>R</td></tr>\n</table>\n\n"
        "<table>\n<tr><td>L</td></tr>\n</table>\n"
    )

