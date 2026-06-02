from torvex_bench.adapters.base import DocumentResult, PageResult, TableResult
from torvex_bench.normalizer import normalize_document, normalize_page, normalize_table


def test_normalize_table_converts_cells_and_bboxes_to_json_safe_values():
    table = TableResult(
        rows=[
            ["A", None, 123],
            ["B", "C", 4.5],
        ],
        bbox_pdfium=[1, 2, 3, 4],
        bbox_plumber=[5, 6, 7, 8],
        bbox_px=[9, 10, 11, 12],
        source="unit_test",
        confidence=0.75,
        metadata={"kind": "borderless"},
    )

    result = normalize_table(table)

    assert result["rows"] == [
        ["A", "", "123"],
        ["B", "C", "4.5"],
    ]

    assert result["bbox_pdfium"] == [1.0, 2.0, 3.0, 4.0]
    assert result["bbox_plumber"] == [5.0, 6.0, 7.0, 8.0]
    assert result["bbox_px"] == [9.0, 10.0, 11.0, 12.0]

    assert all(isinstance(value, float) for value in result["bbox_pdfium"])
    assert all(isinstance(value, float) for value in result["bbox_plumber"])
    assert all(isinstance(value, float) for value in result["bbox_px"])

    assert result["source"] == "unit_test"
    assert result["confidence"] == 0.75
    assert result["metadata"] == {"kind": "borderless"}


def test_normalize_table_allows_missing_optional_bboxes():
    table = TableResult(
        rows=[["A", "B"]],
    )

    result = normalize_table(table)

    assert result["rows"] == [["A", "B"]]
    assert result["bbox_pdfium"] is None
    assert result["bbox_plumber"] is None
    assert result["bbox_px"] is None
    assert result["source"] == "unknown"
    assert result["confidence"] == 1.0
    assert result["metadata"] == {}


def test_normalize_page_normalizes_nested_tables_and_page_fields():
    page = PageResult(
        page_num=2,
        text="hello",
        tables=[
            TableResult(
                rows=[["x", "y"]],
                bbox_pdfium=[1, 2, 3, 4],
                source="table_source",
            )
        ],
        layout_zones=[
            {
                "type": "text",
                "bbox_pdfium": [1, 2, 3, 4],
                "score": 0.9,
            }
        ],
        formula_bboxes=[[1, 2, 3, 4]],
        spotlight_bboxes=[[5, 6, 7, 8]],
        needs_ocr=True,
        ocr_used=True,
        metadata={"page_class": "mixed"},
    )

    result = normalize_page(page)

    assert result["page_num"] == 2
    assert result["text"] == "hello"

    assert result["tables"][0]["rows"] == [["x", "y"]]
    assert result["tables"][0]["bbox_pdfium"] == [1.0, 2.0, 3.0, 4.0]
    assert result["tables"][0]["source"] == "table_source"

    assert result["layout_zones"] == page.layout_zones
    assert result["formula_bboxes"] == [[1.0, 2.0, 3.0, 4.0]]
    assert result["spotlight_bboxes"] == [[5.0, 6.0, 7.0, 8.0]]

    assert result["needs_ocr"] is True
    assert result["ocr_used"] is True
    assert result["metadata"] == {"page_class": "mixed"}    


def test_normalize_document_normalizes_pages_errors_and_metadata():
    document = DocumentResult(
        pdf_path="sample.pdf",
        pages=[
            PageResult(
                page_num=0,
                text="page text",
            )
        ],
        errors=[
            {
                "page_num": 0,
                "message": "test error",
            }
        ],
        metadata={"adapter": "unit_test"},
    )

    result = normalize_document(document)

    assert result["pdf_path"] == "sample.pdf"

    assert len(result["pages"]) == 1
    assert result["pages"][0]["page_num"] == 0
    assert result["pages"][0]["text"] == "page text"

    assert result["errors"] == [
        {
            "page_num": 0,
            "message": "test error",
        }
    ]

    assert result["metadata"] == {"adapter": "unit_test"}