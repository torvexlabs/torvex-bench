from pathlib import Path

from torvex_bench.exporters.olmocr_markdown import (
    export_olmocr_markdown_prediction,
    normalized_document_page_to_markdown,
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
        "text": "Hello olmOCR",
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
        "Hello olmOCR\n\n"
        "<table>\n"
        "<tr><td>Col A</td><td>Col B</td></tr>\n"
        "<tr><td>1</td><td>2</td></tr>\n"
        "</table>\n"
    )


def test_normalized_document_page_to_markdown_matches_zero_based_page() -> None:
    document = {
        "pages": [
            {
                "page_num": 0,
                "text": "Page one",
                "tables": [],
            }
        ]
    }

    markdown = normalized_document_page_to_markdown(document, page=1)

    assert markdown == "Page one\n"


def test_normalized_document_page_to_markdown_matches_one_based_page() -> None:
    document = {
        "pages": [
            {
                "page_num": 2,
                "text": "Page two",
                "tables": [],
            }
        ]
    }

    markdown = normalized_document_page_to_markdown(document, page=2)

    assert markdown == "Page two\n"


def test_normalized_document_page_to_markdown_uses_single_page_fallback() -> None:
    document = {
        "pages": [
            {
                "page_num": 99,
                "text": "Only page",
                "tables": [],
            }
        ]
    }

    markdown = normalized_document_page_to_markdown(document, page=1)

    assert markdown == "Only page\n"


def test_export_olmocr_markdown_prediction_writes_nested_official_path(
    tmp_path: Path,
) -> None:
    document = {
        "pages": [
            {
                "page_num": 0,
                "text": "Prediction text",
                "tables": [],
            }
        ]
    }

    prediction_path = tmp_path / "torvex_extract" / "old_scans" / "1_pg1_repeat1.md"

    result = export_olmocr_markdown_prediction(
        document,
        prediction_path,
        page=1,
    )

    assert result == prediction_path
    assert prediction_path.exists()
    assert prediction_path.read_text(encoding="utf-8") == "Prediction text\n"


def test_olmocr_markdown_emits_display_formula_latex() -> None:
    from torvex_bench.exporters.olmocr_markdown import normalized_page_to_markdown

    page = {
        "text": "Before formula.",
        "formulas": [
            {
                "type": "display_formula",
                "latex": r"\sum_i x_i",
                "status": "accepted",
            }
        ],
        "tables": [],
    }

    markdown = normalized_page_to_markdown(page)

    assert "Before formula." in markdown
    assert "$$\n\\sum_i x_i\n$$" in markdown