from torvex_bench.harnesses.docling_eval_fintabnet import extract_first_table_rows


def test_extract_first_table_rows_from_page_table() -> None:
    normalized = {
        "pages": [
            {
                "tables": [
                    {
                        "rows": [
                            ["A", "B"],
                            ["1", "2"],
                        ]
                    }
                ]
            }
        ]
    }

    assert extract_first_table_rows(normalized) == [["A", "B"], ["1", "2"]]


def test_extract_first_table_rows_returns_empty_when_missing() -> None:
    normalized = {"pages": [{"tables": []}]}

    assert extract_first_table_rows(normalized) == []