"""
FinTabNet table-structure scorer.

This module scores predicted table structures against FinTabNet ground truth.

Ground truth is provided as HTML/restored HTML. Predictions are taken from
adapter `DocumentResult` tables and converted from row lists to HTML before
scoring.

Metrics:
    TEDS        - table structure and cell text similarity
    TEDS-Struct - table structure similarity only

The implementation uses docling-eval's TEDScorer backend:
    docling_eval.evaluators.table.teds.TEDScorer

This module does not perform extraction. It only evaluates extracted tables.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Any

from docling_eval.evaluators.table.teds import TEDScorer
from lxml import html


# FIX: TEDScorer instantiated once at module level, not per sample call.
# Previously score_table_html() called TEDScorer() on every invocation.
# At 1,000 FinTabNet samples that is 1,000 constructor calls. If TEDScorer
# loads any model or compiles anything in __init__, the FinTabNet run timing
# is inflated and latency numbers are wrong. Singleton is the correct pattern
# for a stateless scorer that is called in a tight loop.
_TEDS_SCORER = TEDScorer()


# TableStructureScore stores one sample's result.
# TableStructureSummary stores aggregate run-level results.
@dataclass(frozen=True)
class TableStructureScore:
    """
    Per-sample FinTabNet table structure score.
    """

    sample_id: str
    teds: float
    teds_struct: float
    pred_table_count: int
    gt_rows: int
    gt_cols: int
    pred_rows: int
    pred_cols: int
    has_spans: bool
    backend: str = "docling_eval_teds"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TableStructureSummary:
    """
    Aggregate table structure scores.

    Averaging convention (intentional — do not change):
        mean_teds and mean_teds_struct are computed over ALL samples,
        including those with errors (which contribute 0.0). This penalises
        pipelines that fail silently. samples_error_free is the count of
        samples that scored without any error, reported separately for
        diagnostic purposes but NOT used as the denominator for mean_teds.

    FIX: field was previously named samples_scored which implied the mean
    was computed only over scored samples. Renamed to samples_error_free to
    make the convention unambiguous. The mean is always over samples_total.
    """

    samples_total: int
    samples_error_free: int    # FIX: was samples_scored — misleading name
    mean_teds: float           # sum(teds) / samples_total — errors contribute 0.0
    mean_teds_struct: float    # sum(teds_struct) / samples_total
    missing_table_count: int
    error_count: int
    backend: str = "docling_eval_teds"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Adapters return table rows as Python lists.
# TEDS expects HTML table trees.
# These helpers convert predicted rows into simple HTML.
def rows_to_html(rows: list[list[Any]]) -> str:
    """
    Convert predicted table rows into simple table HTML.

    Adapter rows:
        [["A", "B"], ["1", "2"]]

    HTML:
        <table><tr><td>A</td><td>B</td></tr>...</table>
    """
    html_rows: list[str] = []

    for row in rows or []:
        cells: list[str] = []

        for cell in row or []:
            text = "" if cell is None else str(cell)
            cells.append(f"<td>{escape(text)}</td>")

        html_rows.append("<tr>" + "".join(cells) + "</tr>")

    return "<table>" + "".join(html_rows) + "</table>"


def normalize_table_html(value: Any) -> str:
    """
    Normalize GT/predicted table HTML for docling-eval TEDS.

    FinTabNet GT can be stored as a fragment:
        <tr><td>...</td></tr>

    TEDS expects a parseable HTML table tree, so we wrap fragments in <table>.
    """
    if value is None:
        return ""

    table_html = str(value).strip()

    if not table_html:
        return ""

    lower = table_html.lower()

    if "<table" not in lower and "<tr" in lower:
        table_html = f"<table>{table_html}</table>"

    return table_html


# This is the real metric path.
# It uses the module-level _TEDS_SCORER singleton for both:
#   - TEDS: structure + text
#   - TEDS-Struct: structure only
def score_table_html(
    *,
    gt_html: str,
    pred_html: str,
) -> tuple[float, float]:
    """
    Score one predicted table HTML against one GT table HTML.

    Returns:
        (TEDS, TEDS-Struct)

    Uses the module-level TEDScorer singleton (not a new instance per call).
    """
    gt_html = normalize_table_html(gt_html)
    pred_html = normalize_table_html(pred_html)

    if not gt_html or not pred_html:
        return 0.0, 0.0

    # Re-parse both trees before each scorer call because TEDScorer mutates
    # td/th tags internally. Two separate parse calls is intentional.
    gt_tree = html.fromstring(gt_html)
    pred_tree = html.fromstring(pred_html)

    teds = _TEDS_SCORER(
        gt_table=gt_tree,
        pred_table=pred_tree,
        structure_only=False,
    )

    gt_tree_struct = html.fromstring(gt_html)
    pred_tree_struct = html.fromstring(pred_html)

    teds_struct = _TEDS_SCORER(
        gt_table=gt_tree_struct,
        pred_table=pred_tree_struct,
        structure_only=True,
    )

# 2026-06-06:
# docling-eval TEDS should be normalized, but scorer edge cases can return
# tiny negative or >1.0 float values. Clamp before rounding so benchmark
# JSONL and summary means always stay in the valid [0.0, 1.0] score range.
    return (
    round(max(0.0, min(1.0, float(teds))), 3),
    round(max(0.0, min(1.0, float(teds_struct))), 3),
)


# These helpers pull tables out of DocumentResult-like objects.
# They support both dataclass/object style and dictionary style outputs.
def extract_predicted_tables(document_result: Any) -> list[Any]:
    """
    Extract all tables from a DocumentResult-like object.

    Supports:
    - dataclass/object style: document.pages[0].tables
    - dict style: document["pages"][0]["tables"]
    """
    pages = _get_field(document_result, "pages", default=[])

    tables: list[Any] = []

    for page in pages or []:
        page_tables = _get_field(page, "tables", default=[])
        tables.extend(page_tables or [])

    return tables


def pick_best_predicted_table(tables: list[Any]) -> Any | None:
    """
    FinTabNet sample is one table crop.

    If extractor returns multiple tables, score the largest by cell count.
    """
    if not tables:
        return None

    return max(tables, key=_table_cell_count)


def table_rows(table: Any) -> list[list[Any]]:
    """
    Extract rows from a TableResult-like object.
    """
    rows = _get_field(table, "rows", default=[])

    if not isinstance(rows, list):
        return []

    clean_rows: list[list[Any]] = []

    for row in rows:
        if isinstance(row, list):
            clean_rows.append(row)
        else:
            clean_rows.append([row])

    return clean_rows


# This is the main function runner.py will call.
# It takes one FinTabNetSample and one extraction result, then returns TEDS scores.
def score_fintabnet_sample(
    *,
    sample: Any,
    document_result: Any,
) -> TableStructureScore:
    """
    Score one FinTabNet sample against one extraction result.
    """
    sample_id = str(_get_field(sample, "sample_id", default="unknown"))

    gt_html = str(
        _get_field(sample, "gt_html_restored", default="")
        or _get_field(sample, "gt_html", default="")
    )

    gt_rows = int(_get_field(sample, "rows", default=0) or 0)
    gt_cols = int(_get_field(sample, "cols", default=0) or 0)
    has_spans = bool(_get_field(sample, "has_spans", default=False))

    if not gt_html:
        return TableStructureScore(
            sample_id=sample_id,
            teds=0.0,
            teds_struct=0.0,
            pred_table_count=0,
            gt_rows=gt_rows,
            gt_cols=gt_cols,
            pred_rows=0,
            pred_cols=0,
            has_spans=has_spans,
            error="missing_ground_truth_html",
        )

    predicted_tables = extract_predicted_tables(document_result)
    pred_table = pick_best_predicted_table(predicted_tables)

    if pred_table is None:
        return TableStructureScore(
            sample_id=sample_id,
            teds=0.0,
            teds_struct=0.0,
            pred_table_count=0,
            gt_rows=gt_rows,
            gt_cols=gt_cols,
            pred_rows=0,
            pred_cols=0,
            has_spans=has_spans,
            error="missing_predicted_table",
        )

    pred_rows_list = table_rows(pred_table)
    pred_html = rows_to_html(pred_rows_list)

    pred_rows_count = len(pred_rows_list)
    pred_cols_count = max((len(row) for row in pred_rows_list), default=0)

    try:
        teds, teds_struct = score_table_html(
            gt_html=gt_html,
            pred_html=pred_html,
        )
    except Exception as exc:
        return TableStructureScore(
            sample_id=sample_id,
            teds=0.0,
            teds_struct=0.0,
            pred_table_count=len(predicted_tables),
            gt_rows=gt_rows,
            gt_cols=gt_cols,
            pred_rows=pred_rows_count,
            pred_cols=pred_cols_count,
            has_spans=has_spans,
            error=f"teds_error:{type(exc).__name__}",
        )

    return TableStructureScore(
        sample_id=sample_id,
        teds=teds,
        teds_struct=teds_struct,
        pred_table_count=len(predicted_tables),
        gt_rows=gt_rows,
        gt_cols=gt_cols,
        pred_rows=pred_rows_count,
        pred_cols=pred_cols_count,
        has_spans=has_spans,
        error=None,
    )


# These helpers summarize all per-sample scores into benchmark-level numbers.
def summarize_table_structure_scores(
    scores: list[TableStructureScore],
) -> TableStructureSummary:
    """
    Aggregate per-sample table scores.

    Mean is computed over ALL samples (errors contribute 0.0).
    See TableStructureSummary docstring for the averaging convention.
    """
    if not scores:
        return TableStructureSummary(
            samples_total=0,
            samples_error_free=0,
            mean_teds=0.0,
            mean_teds_struct=0.0,
            missing_table_count=0,
            error_count=0,
        )

    error_free = [score for score in scores if score.error is None]

    return TableStructureSummary(
        samples_total=len(scores),
        samples_error_free=len(error_free),   # FIX: was samples_scored
        mean_teds=round(sum(score.teds for score in scores) / len(scores), 4),
        mean_teds_struct=round(
            sum(score.teds_struct for score in scores) / len(scores),
            4,
        ),
        missing_table_count=sum(
            1 for score in scores if score.error == "missing_predicted_table"
        ),
        error_count=sum(1 for score in scores if score.error is not None),
    )


def save_table_structure_scores_jsonl(
    scores: list[TableStructureScore],
    output_path: str | Path,
) -> None:
    """
    Save per-sample table scores as JSONL.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for score in scores:
            f.write(json.dumps(score.to_dict(), ensure_ascii=False) + "\n")


# Small helpers used only inside this scorer.
def _table_cell_count(table: Any) -> int:
    rows = table_rows(table)
    return sum(len(row or []) for row in rows)


def _get_field(
    obj: Any,
    name: str,
    *,
    default: Any = None,
) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)

    return getattr(obj, name, default)