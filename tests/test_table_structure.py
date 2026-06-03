from torvex_bench.adapters.base import DocumentResult, PageResult, TableResult
from torvex_bench.scorers.table_structure import (
    TableStructureScore,
    extract_predicted_tables,
    rows_to_html,
    score_fintabnet_sample,
    score_table_html,
    summarize_table_structure_scores,
)


class DummySample:
    sample_id = "sample_1"
    gt_html = "<tr><td>A</td><td>B</td></tr>"
    gt_html_restored = "<tr><td>A</td><td>B</td></tr>"
    rows = 1
    cols = 2
    has_spans = False


def test_rows_to_html_escapes_cell_text() -> None:
    html = rows_to_html([["A&B", "<tag>"]])

    assert html == "<table><tr><td>A&amp;B</td><td>&lt;tag&gt;</td></tr></table>"


def test_score_table_html_exact_match_returns_one() -> None:
    teds, teds_struct = score_table_html(
        gt_html="<tr><td>A</td></tr>",
        pred_html="<table><tr><td>A</td></tr></table>",
    )

    assert teds == 1.0
    assert teds_struct == 1.0


def test_score_table_html_text_mismatch_keeps_structure_score() -> None:
    teds, teds_struct = score_table_html(
        gt_html="<tr><td>A</td></tr>",
        pred_html="<table><tr><td>B</td></tr></table>",
    )

    assert teds < 1.0
    assert teds_struct == 1.0


def test_extract_predicted_tables_from_document_result() -> None:
    table = TableResult(rows=[["A"]])
    document = DocumentResult(
        pdf_path="sample.pdf",
        pages=[
            PageResult(page_num=0, tables=[table]),
        ],
    )

    tables = extract_predicted_tables(document)

    assert tables == [table]


def test_score_fintabnet_sample_scores_document_result() -> None:
    document = DocumentResult(
        pdf_path="sample.pdf",
        pages=[
            PageResult(
                page_num=0,
                tables=[
                    TableResult(rows=[["A", "B"]]),
                ],
            )
        ],
    )

    score = score_fintabnet_sample(
        sample=DummySample(),
        document_result=document,
    )

    assert score.sample_id == "sample_1"
    assert score.teds == 1.0
    assert score.teds_struct == 1.0
    assert score.pred_table_count == 1
    assert score.pred_rows == 1
    assert score.pred_cols == 2
    assert score.error is None


def test_score_fintabnet_sample_missing_predicted_table() -> None:
    document = DocumentResult(
        pdf_path="sample.pdf",
        pages=[
            PageResult(page_num=0, tables=[]),
        ],
    )

    score = score_fintabnet_sample(
        sample=DummySample(),
        document_result=document,
    )

    assert score.teds == 0.0
    assert score.teds_struct == 0.0
    assert score.pred_table_count == 0
    assert score.error == "missing_predicted_table"


def test_summarize_table_structure_scores() -> None:
    scores = [
        TableStructureScore(
            sample_id="sample_1",
            teds=1.0,
            teds_struct=1.0,
            pred_table_count=1,
            gt_rows=1,
            gt_cols=1,
            pred_rows=1,
            pred_cols=1,
            has_spans=False,
            error=None,
        ),
        TableStructureScore(
            sample_id="sample_2",
            teds=0.0,
            teds_struct=0.0,
            pred_table_count=0,
            gt_rows=1,
            gt_cols=1,
            pred_rows=0,
            pred_cols=0,
            has_spans=False,
            error="missing_predicted_table",
        ),
    ]

    summary = summarize_table_structure_scores(scores)

    assert summary.samples_total == 2
    assert summary.samples_scored == 1
    assert summary.mean_teds == 0.5
    assert summary.mean_teds_struct == 0.5
    assert summary.missing_table_count == 1
    assert summary.error_count == 1