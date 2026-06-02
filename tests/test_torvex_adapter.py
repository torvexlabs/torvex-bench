import json
from pathlib import Path

from torvex_bench.adapters.base import DocumentResult, PageResult, TableResult
from torvex_bench.adapters.torvex_extract_adapter import (
    TorvexExtractAdapter,
    convert_document,
    convert_page,
    convert_table,
    get_formula_bboxes,
)

SAMPLE_OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "sample_outputs"


def load_sample_output(pattern: str) -> dict:
    matches = sorted(SAMPLE_OUTPUTS_DIR.glob(pattern))
    assert matches, f"No sample output matched pattern: {pattern}"

    return json.loads(matches[0].read_text(encoding="utf-8"))


def test_get_formula_bboxes_uses_direct_formula_bboxes_first():
    raw_page = {
        "formula_bboxes": [
            [1, 2, 3, 4],
            [10, 20, 30, 40],
        ],
        "zones": [
            {
                "type": "display_formula",
                "bbox_pdfium": [100, 200, 300, 400],
            }
        ],
    }

    result = get_formula_bboxes(raw_page)

    assert result == [
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0],
    ]


def test_get_formula_bboxes_falls_back_to_formula_zones():
    raw_page = {
        "zones": [
            {
                "type": "text",
                "bbox_pdfium": [100, 200, 300, 400],
            },
            {
                "type": "display_formula",
                "bbox_pdfium": [1, 2, 3, 4],
            },
            {
                "type": "inline_formula",
                "bbox_pdfium": [10, 20, 30, 40],
            },
            {
                "type": "formula_number",
                "bbox_pdfium": None,
            },
        ]
    }

    result = get_formula_bboxes(raw_page)

    assert result == [
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0],
    ]


def test_get_formula_bboxes_returns_empty_list_when_missing():
    raw_page = {
        "zones": [
            {
                "type": "text",
                "bbox_pdfium": [1, 2, 3, 4],
            }
        ]
    }

    result = get_formula_bboxes(raw_page)

    assert result == []


def test_convert_table_maps_raw_table_to_table_result():
    raw_table = {
        "table_id": "table_0",
        "source": "unit_test_source",
        "bbox_pdfium": [1, 2, 3, 4],
        "bbox_plumber": [5, 6, 7, 8],
        "bbox_px": [9, 10, 11, 12],
        "rows": [
            ["A", None, 123],
            ["B", "C", 4.5],
        ],
        "confidence": 0.8,
        "warnings": ["sample warning"],
    }

    result = convert_table(raw_table)

    assert isinstance(result, TableResult)
    assert result.rows == [
        ["A", "", "123"],
        ["B", "C", "4.5"],
    ]
    assert result.bbox_pdfium == [1.0, 2.0, 3.0, 4.0]
    assert result.bbox_plumber == [5.0, 6.0, 7.0, 8.0]
    assert result.bbox_px == [9.0, 10.0, 11.0, 12.0]
    assert result.source == "unit_test_source"
    assert result.confidence == 0.8
    assert result.metadata == {
        "table_id": "table_0",
        "warnings": ["sample warning"],
    }


def test_convert_page_maps_raw_page_to_page_result():
    raw_page = {
        "page_num": 3,
        "final_text": "hello page",
        "needs_ocr": True,
        "ocr_reason": "empty",
        "page_width": 612,
        "page_height": 792,
        "zones": [
            {
                "type": "text",
                "bbox_pdfium": [1, 2, 3, 4],
                "score": 0.9,
            }
        ],
        "formula_bboxes": [
            [10, 20, 30, 40],
        ],
        "spotlight_bboxes": [
            [50, 60, 70, 80],
        ],
        "tables": [
            {
                "source": "unit_test_table",
                "rows": [["A", "B"]],
                "bbox_pdfium": [1, 2, 3, 4],
                "confidence": 0.9,
            }
        ],
        "metadata": {
            "timings_ms": {
                "page_total": 123.4,
            }
        },
        "layout_grade": "EXCELLENT",
        "page_class": "mixed",
    }

    result = convert_page(raw_page)

    assert isinstance(result, PageResult)
    assert result.page_num == 3
    assert result.text == "hello page"

    assert len(result.tables) == 1
    assert result.tables[0].rows == [["A", "B"]]
    assert result.tables[0].bbox_pdfium == [1.0, 2.0, 3.0, 4.0]
    assert result.tables[0].source == "unit_test_table"

    assert result.layout_zones == raw_page["zones"]
    assert result.formula_bboxes == [[10.0, 20.0, 30.0, 40.0]]
    assert result.spotlight_bboxes == [[50.0, 60.0, 70.0, 80.0]]

    assert result.needs_ocr is True
    assert result.ocr_used is True

    assert result.metadata["ocr_reason"] == "empty"
    assert result.metadata["page_width"] == 612
    assert result.metadata["page_height"] == 792
    assert result.metadata["layout_grade"] == "EXCELLENT"
    assert result.metadata["page_class"] == "mixed"
    assert result.metadata["timings_ms"]["page_total"] == 123.4


def test_convert_document_maps_raw_output_to_document_result():
    raw_output = {
        "pdf": "sample.pdf",
        "errors": [
            {
                "page_num": 0,
                "message": "sample error",
            }
        ],
        "pages": [
            {
                "page_num": 0,
                "final_text": "page text",
                "tables": [],
                "zones": [],
            }
        ],
        "metadata": {
            "adapter": "unit_test",
        },
    }

    result = convert_document(raw_output)

    assert isinstance(result, DocumentResult)
    assert result.pdf_path == "sample.pdf"
    assert len(result.pages) == 1
    assert result.pages[0].page_num == 0
    assert result.pages[0].text == "page text"
    assert result.errors == [
        {
            "page_num": 0,
            "message": "sample error",
        }
    ]
    assert result.metadata == {"adapter": "unit_test"}


def test_convert_document_reads_aapl_sample_output():
    raw_output = load_sample_output("01_AAPL*_sample_output.json")

    result = convert_document(raw_output)

    assert isinstance(result, DocumentResult)
    assert result.pdf_path == raw_output["pdf"]
    assert len(result.pages) == len(raw_output["pages"])

    first_raw_page = raw_output["pages"][0]
    first_page = result.pages[0]

    assert first_page.page_num == first_raw_page["page_num"]
    assert first_page.text == first_raw_page["final_text"]
    assert first_page.needs_ocr is bool(first_raw_page["needs_ocr"])
    assert first_page.layout_zones == first_raw_page["zones"]
    assert first_page.metadata["ocr_reason"] == first_raw_page["ocr_reason"]


def test_convert_document_reads_invoice_sample_output():
    raw_output = load_sample_output("05_InvoiceBatch*_sample_output.json")

    result = convert_document(raw_output)

    assert isinstance(result, DocumentResult)
    assert result.pdf_path == raw_output["pdf"]
    assert len(result.pages) == len(raw_output["pages"])

    first_page = result.pages[0]

    assert first_page.needs_ocr is True
    assert first_page.ocr_used is True
    assert len(first_page.tables) >= 1

    first_table = first_page.tables[0]

    assert first_table.rows
    assert first_table.bbox_pdfium is not None
    assert first_table.bbox_plumber is not None
    assert first_table.bbox_px is not None
    assert first_table.source != "unknown"


def test_torvex_extract_adapter_imports():
    adapter = TorvexExtractAdapter()

    assert adapter.name == "torvex_extract"
    assert adapter.version == "0.1.0"
    assert callable(adapter.extract)
    assert callable(adapter.extract_document)