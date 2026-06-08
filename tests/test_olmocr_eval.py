from pathlib import Path

from torvex_bench.adapters.base import DocumentResult, PageResult, TableResult
from torvex_bench.datasets.olmocr import OlmOCRBenchSample
from torvex_bench.harnesses.olmocr_eval import (
    generate_olmocr_predictions_from_samples,
    prediction_path_for_sample_page,
)


class FakeAdapter:
    def extract_document(self, pdf_path: str | Path) -> DocumentResult:
        return DocumentResult(
            pdf_path=str(pdf_path),
            pages=[
                PageResult(
                    page_num=0,
                    text="Hello olmOCR",
                    tables=[
                        TableResult(
                            rows=[
                                ["A", "B"],
                                ["1", "2"],
                            ]
                        )
                    ],
                )
            ],
            errors=[],
            metadata={"adapter": "fake"},
        )


class FailingAdapter:
    def extract_document(self, pdf_path: str | Path) -> DocumentResult:
        raise RuntimeError("boom")


def make_sample(tmp_path: Path) -> OlmOCRBenchSample:
    pdf_path = tmp_path / "bench_data" / "pdfs" / "old_scans" / "1.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4 fake\n")

    return OlmOCRBenchSample(
        sample_id="olmocr_test",
        pdf="old_scans/1.pdf",
        local_pdf_path=pdf_path,
        pages=[1],
        source_jsonls=["old_scans.jsonl"],
        metadata={},
    )


def test_prediction_path_for_sample_page_uses_official_pattern(tmp_path: Path) -> None:
    sample = make_sample(tmp_path)

    path = prediction_path_for_sample_page(
        sample=sample,
        prediction_dir=tmp_path / "torvex_extract",
        page=1,
    )

    assert path == tmp_path / "torvex_extract" / "old_scans" / "1_pg1_repeat1.md"


def test_generate_predictions_from_samples_writes_prediction_and_artifacts(
    tmp_path: Path,
) -> None:
    sample = make_sample(tmp_path)

    summary = generate_olmocr_predictions_from_samples(
        samples=[sample],
        prediction_dir=tmp_path / "bench_data" / "torvex_extract",
        adapter=FakeAdapter(),
        overwrite=True,
        save_raw=True,
        raw_dir=tmp_path / "bench_data" / "raw_outputs" / "torvex_extract",
        save_normalized=True,
        normalized_dir=tmp_path / "bench_data" / "normalized" / "torvex_extract",
    )

    prediction_path = (
        tmp_path
        / "bench_data"
        / "torvex_extract"
        / "old_scans"
        / "1_pg1_repeat1.md"
    )
    raw_path = tmp_path / "bench_data" / "raw_outputs" / "torvex_extract" / "olmocr_test.json"
    normalized_path = (
        tmp_path
        / "bench_data"
        / "normalized"
        / "torvex_extract"
        / "olmocr_test.json"
    )

    assert summary.requested == 1
    assert summary.processed == 1
    assert summary.predictions_written == 1
    assert summary.empty_predictions_written == 0
    assert summary.errors == 0

    assert prediction_path.exists()
    assert raw_path.exists()
    assert normalized_path.exists()

    markdown = prediction_path.read_text(encoding="utf-8")
    assert "Hello olmOCR" in markdown
    assert "<table>" in markdown
    assert "<td>A</td>" in markdown


def test_generate_predictions_skips_existing_when_not_overwrite(tmp_path: Path) -> None:
    sample = make_sample(tmp_path)
    prediction_dir = tmp_path / "bench_data" / "torvex_extract"
    prediction_path = prediction_dir / "old_scans" / "1_pg1_repeat1.md"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text("existing\n", encoding="utf-8")

    summary = generate_olmocr_predictions_from_samples(
        samples=[sample],
        prediction_dir=prediction_dir,
        adapter=FakeAdapter(),
        overwrite=False,
    )

    assert summary.requested == 1
    assert summary.processed == 0
    assert summary.skipped_existing == 1
    assert prediction_path.read_text(encoding="utf-8") == "existing\n"


def test_generate_predictions_writes_empty_prediction_on_error(tmp_path: Path) -> None:
    sample = make_sample(tmp_path)

    summary = generate_olmocr_predictions_from_samples(
        samples=[sample],
        prediction_dir=tmp_path / "bench_data" / "torvex_extract",
        adapter=FailingAdapter(),
        overwrite=True,
    )

    prediction_path = (
        tmp_path
        / "bench_data"
        / "torvex_extract"
        / "old_scans"
        / "1_pg1_repeat1.md"
    )
    error_path = (
        tmp_path
        / "bench_data"
        / "errors"
        / "torvex_extract"
        / "olmocr_test.error.json"
    )

    assert summary.requested == 1
    assert summary.processed == 1
    assert summary.predictions_written == 0
    assert summary.empty_predictions_written == 1
    assert summary.errors == 1

    assert prediction_path.exists()
    assert prediction_path.read_text(encoding="utf-8") == ""
    assert error_path.exists()


def test_generate_predictions_writes_one_prediction_per_page(tmp_path: Path) -> None:
    pdf_path = tmp_path / "bench_data" / "pdfs" / "multi_page" / "sample.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4 fake\n")

    sample = OlmOCRBenchSample(
        sample_id="olmocr_multi",
        pdf="multi_page/sample.pdf",
        local_pdf_path=pdf_path,
        pages=[1, 2],
        source_jsonls=["multi_column.jsonl"],
        metadata={},
    )

    summary = generate_olmocr_predictions_from_samples(
        samples=[sample],
        prediction_dir=tmp_path / "bench_data" / "torvex_extract",
        adapter=FakeAdapter(),
        overwrite=True,
    )

    assert summary.predictions_written == 2

    assert (
        tmp_path
        / "bench_data"
        / "torvex_extract"
        / "multi_page"
        / "sample_pg1_repeat1.md"
    ).exists()

    assert (
        tmp_path
        / "bench_data"
        / "torvex_extract"
        / "multi_page"
        / "sample_pg2_repeat1.md"
    ).exists()