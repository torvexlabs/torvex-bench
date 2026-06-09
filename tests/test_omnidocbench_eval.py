from pathlib import Path

from PIL import Image

from torvex_bench.adapters.base import DocumentResult, PageResult, TableResult
from torvex_bench.datasets.omnidocbench import OmniDocBenchSample
from torvex_bench.harnesses.omnidocbench_eval import (
    generate_omnidocbench_predictions_from_samples,
    image_to_scanned_pdf,
)


class FakeAdapter:
    def extract_document(self, pdf_path: str | Path) -> DocumentResult:
        return DocumentResult(
            pdf_path=str(pdf_path),
            pages=[
                PageResult(
                    page_num=0,
                    text="Hello Omni",
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


def make_sample(tmp_path: Path) -> OmniDocBenchSample:
    image_path = tmp_path / "images" / "page-test.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 40), "white").save(image_path)

    return OmniDocBenchSample(
        source_index=0,
        sample_id="omnidocbench_000000_test",
        image_filename="page-test.png",
        image_repo_path="images/page-test.png",
        image_path=image_path,
        page_info={
            "image_path": "page-test.png",
            "width": 80,
            "height": 40,
        },
        layout_dets=[],
        extra={},
        metadata={},
    )


def test_image_to_scanned_pdf_creates_pdf(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    pdf_path = tmp_path / "page.pdf"

    Image.new("RGB", (80, 40), "white").save(image_path)

    result = image_to_scanned_pdf(image_path, pdf_path)

    assert result == pdf_path
    assert pdf_path.exists()
    assert pdf_path.read_bytes().startswith(b"%PDF")


def test_generate_predictions_from_samples_writes_prediction_and_artifacts(
    tmp_path: Path,
) -> None:
    sample = make_sample(tmp_path)

    summary = generate_omnidocbench_predictions_from_samples(
        samples=[sample],
        prediction_dir=tmp_path / "predictions",
        temp_pdfs_dir=tmp_path / "temp_pdfs",
        adapter=FakeAdapter(),
        overwrite=True,
        save_raw=True,
        raw_dir=tmp_path / "raw_outputs",
        save_normalized=True,
        normalized_dir=tmp_path / "normalized",
    )

    prediction_path = tmp_path / "predictions" / "page-test.md"
    temp_pdf_path = tmp_path / "temp_pdfs" / "page-test.pdf"
    raw_path = tmp_path / "raw_outputs" / "page-test.json"
    normalized_path = tmp_path / "normalized" / "page-test.json"

    assert summary.requested == 1
    assert summary.processed == 1
    assert summary.predictions_written == 1
    assert summary.empty_predictions_written == 0
    assert summary.errors == 0
    assert prediction_path.exists()
    assert temp_pdf_path.exists()
    assert raw_path.exists()
    assert normalized_path.exists()

    markdown = prediction_path.read_text(encoding="utf-8")
    assert "Hello Omni" in markdown
    assert "<table>" in markdown
    assert "<td>A</td>" in markdown


def test_generate_predictions_skips_existing_when_not_overwrite(
    tmp_path: Path,
) -> None:
    sample = make_sample(tmp_path)
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir(parents=True)
    (prediction_dir / "page-test.md").write_text("existing\n", encoding="utf-8")

    summary = generate_omnidocbench_predictions_from_samples(
        samples=[sample],
        prediction_dir=prediction_dir,
        temp_pdfs_dir=tmp_path / "temp_pdfs",
        adapter=FakeAdapter(),
        overwrite=False,
    )

    assert summary.requested == 1
    assert summary.processed == 0
    assert summary.skipped_existing == 1
    assert (prediction_dir / "page-test.md").read_text(encoding="utf-8") == "existing\n"


def test_generate_predictions_writes_empty_prediction_on_error(
    tmp_path: Path,
) -> None:
    sample = make_sample(tmp_path)

    summary = generate_omnidocbench_predictions_from_samples(
        samples=[sample],
        prediction_dir=tmp_path / "predictions",
        temp_pdfs_dir=tmp_path / "temp_pdfs",
        adapter=FailingAdapter(),
        overwrite=True,
        save_raw=True,
        raw_dir=tmp_path / "raw_outputs",
    )

    prediction_path = tmp_path / "predictions" / "page-test.md"
    error_path = tmp_path / "raw_outputs" / "page-test.error.json"

    assert summary.requested == 1
    assert summary.processed == 1
    assert summary.predictions_written == 0
    assert summary.empty_predictions_written == 1
    assert summary.errors == 1
    assert prediction_path.exists()
    assert prediction_path.read_text(encoding="utf-8") == ""
    assert error_path.exists()


def test_omnidocbench_prediction_summary_can_record_formula_enabled(tmp_path) -> None:
    from torvex_bench.harnesses.omnidocbench_eval import OmniDocBenchPredictionSummary

    summary = OmniDocBenchPredictionSummary(
        requested=1,
        processed=1,
        predictions_written=1,
        empty_predictions_written=0,
        skipped_existing=0,
        errors=0,
        prediction_dir=tmp_path / "predictions",
        temp_pdfs_dir=tmp_path / "temp_pdfs",
        formula_enabled=True,
    )

    assert summary.formula_enabled is True