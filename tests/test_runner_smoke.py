from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from torvex_bench.adapters.base import (
    DocumentResult,
    ExtractionAdapter,
    PageResult,
    TableResult,
)
from torvex_bench.runner import (
    adapter_name,
    adapter_version,
    count_successes,
    document_result_to_dict,
    make_output_paths,
    resolve_sample_id,
    resolve_sample_pdf_path,
    run_one_sample,
    run_samples,
    safe_name,
    sample_to_metadata,
    summarize_normalized_document,
)


class FakeAdapter(ExtractionAdapter):
    name = "fake_engine"
    version = "0.1-test"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def extract_document(self, pdf_path: str) -> DocumentResult:
        self.calls.append(pdf_path)

        return DocumentResult(
            pdf_path=pdf_path,
            pages=[
                PageResult(
                    page_num=0,
                    text="hello page",
                    tables=[
                        TableResult(
                            rows=[["A", "B"], ["1", "2"]],
                            bbox_pdfium=[1, 2, 3, 4],
                            bbox_plumber=[5, 6, 7, 8],
                            bbox_px=[9, 10, 11, 12],
                            source="fake_table",
                            confidence=0.9,
                            metadata={"kind": "unit"},
                        )
                    ],
                    layout_zones=[
                        {
                            "type": "text",
                            "bbox_pdfium": [1, 2, 3, 4],
                            "score": 0.99,
                        }
                    ],
                    formula_bboxes=[[10, 20, 30, 40]],
                    spotlight_bboxes=[[50, 60, 70, 80]],
                    needs_ocr=True,
                    ocr_used=True,
                    metadata={"page_class": "mixed"},
                )
            ],
            errors=[],
            metadata={"adapter": "fake"},
        )


class FailingAdapter(ExtractionAdapter):
    name = "failing_engine"
    version = "0.1-test"

    def extract_document(self, pdf_path: str) -> DocumentResult:
        raise RuntimeError(f"boom: {pdf_path}")


@dataclass
class SimpleSample:
    sample_id: str
    pdf_path: Path
    gt_html: str = "<table>huge gt</table>"
    gt_bboxes: list[list[float]] | None = None
    tests: list[dict] | None = None
    keep_me: str = "metadata"


@dataclass
class OmniLikeSample:
    sample_id: str
    pdf_path_digital: Path
    pdf_path_scanned: Path
    gt_layout_dets: list[dict]
    gt_text_blocks: list[dict]
    keep_me: str = "omni"

    def pdf_path(self, input_type: str = "digital") -> Path:
        if input_type == "scanned":
            return self.pdf_path_scanned

        return self.pdf_path_digital


def make_pdf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4 fake test pdf\n")
    return path


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_safe_name_makes_filesystem_safe_names() -> None:
    assert safe_name("hello world") == "hello_world"
    assert safe_name("a/b:c*d?e") == "a_b_c_d_e"
    assert safe_name("   ") == "sample"


def test_adapter_name_and_version() -> None:
    adapter = FakeAdapter()

    assert adapter_name(adapter) == "fake_engine"
    assert adapter_version(adapter) == "0.1-test"


def test_make_output_paths_are_deterministic(tmp_path: Path) -> None:
    paths = make_output_paths(
        output_dir=tmp_path,
        dataset="olmocr bench",
        engine="fake/engine",
        sample_id="sample:001",
    )

    base = tmp_path / "olmocr_bench" / "fake_engine"

    assert paths["raw"] == base / "raw" / "sample_001.json"
    assert paths["normalized"] == base / "normalized" / "sample_001.json"
    assert paths["record"] == base / "records" / "sample_001.json"
    assert paths["jsonl"] == base / "run_records.jsonl"


def test_summarize_normalized_document_counts_pages_tables_and_formula_boxes() -> None:
    normalized = {
        "pages": [
            {
                "tables": [{}, {}],
                "formula_bboxes": [[1, 2, 3, 4]],
            },
            {
                "tables": [{}],
                "formula_bboxes": [[5, 6, 7, 8], [9, 10, 11, 12]],
            },
        ]
    }

    assert summarize_normalized_document(normalized) == {
        "page_count": 2,
        "table_count": 3,
        "formula_bbox_count": 3,
    }


def test_document_result_to_dict_is_json_safe(tmp_path: Path) -> None:
    pdf_path = make_pdf(tmp_path / "sample.pdf")
    adapter = FakeAdapter()
    document = adapter.extract_document(str(pdf_path))

    result = document_result_to_dict(document)

    assert result["pdf_path"] == str(pdf_path)
    assert result["pages"][0]["text"] == "hello page"
    assert result["pages"][0]["tables"][0]["rows"] == [["A", "B"], ["1", "2"]]
    assert result["metadata"] == {"adapter": "fake"}


def test_run_one_sample_success_writes_raw_normalized_record_and_jsonl(
    tmp_path: Path,
) -> None:
    pdf_path = make_pdf(tmp_path / "inputs" / "sample.pdf")
    output_dir = tmp_path / "results"
    adapter = FakeAdapter()

    record = run_one_sample(
        adapter=adapter,
        pdf_path=pdf_path,
        sample_id="sample_001",
        dataset="unit_dataset",
        output_dir=output_dir,
        input_type="scanned",
        sample_metadata={"source": "unit"},
        run_metadata={"device": "cpu"},
    )

    assert record.status == "ok"
    assert record.sample_id == "sample_001"
    assert record.dataset == "unit_dataset"
    assert record.engine == "fake_engine"
    assert record.engine_version == "0.1-test"
    assert record.pdf_path == str(pdf_path)
    assert record.input_type == "scanned"

    assert record.page_count == 1
    assert record.table_count == 1
    assert record.formula_bbox_count == 1
    assert record.error_type is None
    assert record.error_message is None

    assert adapter.calls == [str(pdf_path)]

    raw_path = Path(record.raw_output_path or "")
    normalized_path = Path(record.normalized_output_path or "")
    record_path = Path(record.record_output_path or "")
    jsonl_path = output_dir / "unit_dataset" / "fake_engine" / "run_records.jsonl"

    assert raw_path.exists()
    assert normalized_path.exists()
    assert record_path.exists()
    assert jsonl_path.exists()

    raw = read_json(raw_path)
    normalized = read_json(normalized_path)
    saved_record = read_json(record_path)

    assert raw["pages"][0]["text"] == "hello page"
    assert raw["pages"][0]["tables"][0]["rows"] == [["A", "B"], ["1", "2"]]

    assert normalized["pages"][0]["text"] == "hello page"
    assert normalized["pages"][0]["tables"][0]["bbox_pdfium"] == [1.0, 2.0, 3.0, 4.0]
    assert normalized["pages"][0]["formula_bboxes"] == [[10.0, 20.0, 30.0, 40.0]]

    assert saved_record["status"] == "ok"
    assert saved_record["input_type"] == "scanned"
    assert saved_record["metadata"]["sample"] == {"source": "unit"}
    assert saved_record["metadata"]["run"] == {"device": "cpu"}

    jsonl_records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(jsonl_records) == 1
    assert jsonl_records[0]["sample_id"] == "sample_001"
    assert jsonl_records[0]["status"] == "ok"


def test_run_one_sample_missing_pdf_records_error(tmp_path: Path) -> None:
    missing_pdf = tmp_path / "missing.pdf"
    output_dir = tmp_path / "results"

    record = run_one_sample(
        adapter=FakeAdapter(),
        pdf_path=missing_pdf,
        sample_id="missing",
        dataset="unit_dataset",
        output_dir=output_dir,
    )

    assert record.status == "error"
    assert record.error_type == "FileNotFoundError"
    assert "PDF not found" in str(record.error_message)
    assert record.raw_output_path is None
    assert record.normalized_output_path is None

    record_path = Path(record.record_output_path or "")
    jsonl_path = output_dir / "unit_dataset" / "fake_engine" / "run_records.jsonl"

    assert record_path.exists()
    assert jsonl_path.exists()

    saved_record = read_json(record_path)
    assert saved_record["status"] == "error"
    assert saved_record["error_type"] == "FileNotFoundError"


def test_run_one_sample_adapter_error_records_error(tmp_path: Path) -> None:
    pdf_path = make_pdf(tmp_path / "sample.pdf")
    output_dir = tmp_path / "results"

    record = run_one_sample(
        adapter=FailingAdapter(),
        pdf_path=pdf_path,
        sample_id="boom",
        dataset="unit_dataset",
        output_dir=output_dir,
    )

    assert record.status == "error"
    assert record.error_type == "RuntimeError"
    assert "boom" in str(record.error_message)
    assert record.traceback is not None
    assert "RuntimeError" in record.traceback

    record_path = Path(record.record_output_path or "")
    assert record_path.exists()

    saved_record = read_json(record_path)
    assert saved_record["status"] == "error"
    assert saved_record["error_type"] == "RuntimeError"


def test_run_one_sample_raise_on_error_reraises_after_writing_record(
    tmp_path: Path,
) -> None:
    pdf_path = make_pdf(tmp_path / "sample.pdf")
    output_dir = tmp_path / "results"

    with pytest.raises(RuntimeError):
        run_one_sample(
            adapter=FailingAdapter(),
            pdf_path=pdf_path,
            sample_id="boom",
            dataset="unit_dataset",
            output_dir=output_dir,
            raise_on_error=True,
        )

    record_path = (
        output_dir
        / "unit_dataset"
        / "failing_engine"
        / "records"
        / "boom.json"
    )

    assert record_path.exists()
    assert read_json(record_path)["status"] == "error"


def test_run_one_sample_overwrite_false_raises_when_outputs_exist(
    tmp_path: Path,
) -> None:
    pdf_path = make_pdf(tmp_path / "sample.pdf")
    output_dir = tmp_path / "results"

    run_one_sample(
        adapter=FakeAdapter(),
        pdf_path=pdf_path,
        sample_id="sample_001",
        dataset="unit_dataset",
        output_dir=output_dir,
    )

    with pytest.raises(FileExistsError):
        run_one_sample(
            adapter=FakeAdapter(),
            pdf_path=pdf_path,
            sample_id="sample_001",
            dataset="unit_dataset",
            output_dir=output_dir,
            overwrite=False,
        )


def test_resolve_sample_pdf_path_from_mapping() -> None:
    assert resolve_sample_pdf_path(
        {"pdf_path": "plain.pdf"}
    ) == Path("plain.pdf")

    assert resolve_sample_pdf_path(
        {
            "pdf_path_digital": "digital.pdf",
            "pdf_path_scanned": "scanned.pdf",
        },
        input_type="digital",
    ) == Path("digital.pdf")

    assert resolve_sample_pdf_path(
        {
            "pdf_path_digital": "digital.pdf",
            "pdf_path_scanned": "scanned.pdf",
        },
        input_type="scanned",
    ) == Path("scanned.pdf")

    with pytest.raises(KeyError):
        resolve_sample_pdf_path({"sample_id": "no_path"})


def test_resolve_sample_pdf_path_from_object_field(tmp_path: Path) -> None:
    sample = SimpleSample(
        sample_id="simple",
        pdf_path=tmp_path / "simple.pdf",
    )

    assert resolve_sample_pdf_path(sample) == tmp_path / "simple.pdf"


def test_resolve_sample_pdf_path_from_omnidocbench_style_method(
    tmp_path: Path,
) -> None:
    sample = OmniLikeSample(
        sample_id="omni",
        pdf_path_digital=tmp_path / "digital.pdf",
        pdf_path_scanned=tmp_path / "scanned.pdf",
        gt_layout_dets=[{"huge": True}],
        gt_text_blocks=[{"huge": True}],
    )

    assert resolve_sample_pdf_path(sample, input_type="digital") == tmp_path / "digital.pdf"
    assert resolve_sample_pdf_path(sample, input_type="scanned") == tmp_path / "scanned.pdf"


def test_resolve_sample_id_from_mapping_object_and_fallback(tmp_path: Path) -> None:
    assert resolve_sample_id({"sample_id": "abc"}, fallback_index=1) == "abc"
    assert resolve_sample_id({}, fallback_index=7) == "sample_000007"

    sample = SimpleSample(
        sample_id="simple",
        pdf_path=tmp_path / "sample.pdf",
    )

    assert resolve_sample_id(sample, fallback_index=9) == "simple"

    class NoId:
        pass

    assert resolve_sample_id(NoId(), fallback_index=3) == "sample_000003"


def test_sample_to_metadata_strips_heavy_gt_fields(tmp_path: Path) -> None:
    sample = SimpleSample(
        sample_id="simple",
        pdf_path=tmp_path / "simple.pdf",
        gt_html="<table>big</table>",
        gt_bboxes=[[1, 2, 3, 4]],
        tests=[{"id": "test"}],
        keep_me="yes",
    )

    metadata = sample_to_metadata(sample)

    assert metadata["sample_id"] == "simple"
    assert metadata["pdf_path"] == str(tmp_path / "simple.pdf")
    assert metadata["keep_me"] == "yes"

    assert "gt_html" not in metadata
    assert "gt_bboxes" not in metadata
    assert "tests" not in metadata


def test_sample_to_metadata_strips_omnidocbench_gt_fields(tmp_path: Path) -> None:
    sample = OmniLikeSample(
        sample_id="omni",
        pdf_path_digital=tmp_path / "digital.pdf",
        pdf_path_scanned=tmp_path / "scanned.pdf",
        gt_layout_dets=[{"huge": True}],
        gt_text_blocks=[{"huge": True}],
        keep_me="yes",
    )

    metadata = sample_to_metadata(sample)

    assert metadata["sample_id"] == "omni"
    assert metadata["pdf_path_digital"] == str(tmp_path / "digital.pdf")
    assert metadata["pdf_path_scanned"] == str(tmp_path / "scanned.pdf")
    assert metadata["keep_me"] == "yes"

    assert "gt_layout_dets" not in metadata
    assert "gt_text_blocks" not in metadata


def test_run_samples_runs_many_samples_and_passes_input_type(
    tmp_path: Path,
) -> None:
    digital_pdf = make_pdf(tmp_path / "digital.pdf")
    scanned_pdf = make_pdf(tmp_path / "scanned.pdf")

    sample = OmniLikeSample(
        sample_id="omni_sample",
        pdf_path_digital=digital_pdf,
        pdf_path_scanned=scanned_pdf,
        gt_layout_dets=[{"huge": True}],
        gt_text_blocks=[{"huge": True}],
    )

    adapter = FakeAdapter()

    records = run_samples(
        adapter=adapter,
        samples=[sample],
        dataset="omnidocbench",
        output_dir=tmp_path / "results",
        input_type="scanned",
        run_metadata={"device": "cpu"},
    )

    assert len(records) == 1
    assert records[0].status == "ok"
    assert records[0].input_type == "scanned"
    assert records[0].pdf_path == str(scanned_pdf)
    assert adapter.calls == [str(scanned_pdf)]

    saved_record = read_json(records[0].record_output_path or "")
    assert saved_record["input_type"] == "scanned"
    assert saved_record["metadata"]["run"] == {"device": "cpu"}
    assert "gt_layout_dets" not in saved_record["metadata"]["sample"]


def test_run_samples_limit(tmp_path: Path) -> None:
    pdf1 = make_pdf(tmp_path / "one.pdf")
    pdf2 = make_pdf(tmp_path / "two.pdf")

    samples = [
        {"sample_id": "one", "pdf_path": str(pdf1)},
        {"sample_id": "two", "pdf_path": str(pdf2)},
    ]

    adapter = FakeAdapter()

    records = run_samples(
        adapter=adapter,
        samples=samples,
        dataset="unit_dataset",
        output_dir=tmp_path / "results",
        limit=1,
    )

    assert len(records) == 1
    assert records[0].sample_id == "one"
    assert adapter.calls == [str(pdf1)]


def test_count_successes() -> None:
    ok_record = type("Record", (), {"status": "ok"})()
    error_record = type("Record", (), {"status": "error"})()

    assert count_successes([ok_record, error_record, ok_record]) == {
        "ok": 2,
        "error": 1,
        "total": 3,
    }