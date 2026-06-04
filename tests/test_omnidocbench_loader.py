import json
from pathlib import Path

import pytest

from torvex_bench.datasets.omnidocbench import (
    INPUT_TYPE_DIGITAL,
    INPUT_TYPE_SCANNED,
    OmniDocBenchSample,
    iter_omnidocbench_samples_from_manifest,
    load_gt_json,
    make_sample_id,
    materialize_omnidocbench_sample,
    poly_to_xyxy,
    prepare_omnidocbench,
    sample_from_manifest_record,
    save_manifest,
)


FAKE_PDF_BYTES = b"%PDF-1.4\n% fake omnidocbench pdf\n%%EOF"


def make_fake_raw_record() -> dict:
    return {
        "page_info": {
            "image_path": "page-test-001.png",
            "width": 1000,
            "height": 2000,
            "page_attribute": {
                "data_source": "book",
                "language": "english",
                "layout": "single_column",
                "special_issue": ["fuzzy_scan"],
            },
        },
        "layout_dets": [
            {
                "category_type": "text_block",
                "poly": [10, 20, 110, 20, 110, 70, 10, 70],
                "ignore": False,
                "order": 2,
                "anno_id": "box_text_1",
                "attribute": {},
                "text": "hello world",
            },
            {
                "category_type": "table",
                "poly": [20, 100, 220, 100, 220, 300, 20, 300],
                "ignore": False,
                "order": 3,
                "anno_id": "box_table_1",
                "attribute": {},
                "html": "<table><tr><td>A</td></tr></table>",
            },
            {
                "category_type": "equation_isolated",
                "poly": [30, 400, 230, 400, 230, 500, 30, 500],
                "ignore": False,
                "order": 4,
                "anno_id": "box_formula_1",
                "attribute": {},
                "latex": "$$x+y$$",
            },
            {
                "category_type": "abandon",
                "poly": [0, 0, 100, 0, 100, 20, 0, 20],
                "ignore": True,
                "order": 1,
                "anno_id": "box_ignore_1",
                "attribute": {},
            },
        ],
        "extra": {},
    }


def create_fake_pdf_folders(raw_data_dir: Path) -> None:
    pdfs_dir = raw_data_dir / "pdfs"
    ori_pdfs_dir = raw_data_dir / "ori_pdfs"

    pdfs_dir.mkdir(parents=True, exist_ok=True)
    ori_pdfs_dir.mkdir(parents=True, exist_ok=True)

    (pdfs_dir / "page-test-001.pdf").write_bytes(FAKE_PDF_BYTES)
    (ori_pdfs_dir / "page-test-001.pdf").write_bytes(FAKE_PDF_BYTES)


def test_make_sample_id_is_stable() -> None:
    sample_id_1 = make_sample_id(
        source_index=7,
        image_filename="page-test-001.png",
    )

    sample_id_2 = make_sample_id(
        source_index=7,
        image_filename="page-test-001.png",
    )

    assert sample_id_1 == sample_id_2
    assert sample_id_1.startswith("omnidocbench_000007_")


def test_poly_to_xyxy_converts_quad_to_bbox() -> None:
    result = poly_to_xyxy([10, 20, 110, 20, 110, 70, 10, 70])

    assert result == [10, 20, 110, 70]


def test_poly_to_xyxy_rejects_invalid_poly() -> None:
    with pytest.raises(ValueError):
        poly_to_xyxy([1, 2, 3, 4])


def test_materialize_omnidocbench_sample_builds_sample(tmp_path: Path) -> None:
    raw_data_dir = tmp_path / "raw"
    create_fake_pdf_folders(raw_data_dir)

    sample = materialize_omnidocbench_sample(
        raw_record=make_fake_raw_record(),
        source_index=0,
        raw_data_dir=raw_data_dir,
        output_dir=tmp_path / "out",
    )

    assert isinstance(sample, OmniDocBenchSample)
    assert sample.source_index == 0
    assert sample.image_filename == "page-test-001.png"

    assert sample.pdf_path_digital == raw_data_dir / "ori_pdfs" / "page-test-001.pdf"
    assert sample.pdf_path_scanned == raw_data_dir / "pdfs" / "page-test-001.pdf"

    assert sample.pdf_path(INPUT_TYPE_DIGITAL) == sample.pdf_path_digital
    assert sample.pdf_path(INPUT_TYPE_SCANNED) == sample.pdf_path_scanned

    assert sample.page_width == 1000.0
    assert sample.page_height == 2000.0

    assert sample.data_source == "book"
    assert sample.language == "english"
    assert sample.layout == "single_column"
    assert sample.special_issues == ["fuzzy_scan"]

    assert sample.has_table is True
    assert sample.has_formula is True
    assert sample.has_figure is False

    assert len(sample.gt_layout_dets) == 4
    assert len(sample.gt_text_blocks) == 1
    assert len(sample.gt_tables) == 1

    assert [det["anno_id"] for det in sample.gt_reading_order] == [
        "box_text_1",
        "box_table_1",
        "box_formula_1",
    ]

    assert sample.gt_bboxes_xyxy[0] == [10.0, 20.0, 110.0, 70.0]
    assert sample.gt_bboxes_raw_poly[0] == [10.0, 20.0, 110.0, 20.0, 110.0, 70.0, 10.0, 70.0]

    assert sample.metadata["dataset_slug"] == "opendatalab/OmniDocBench"
    assert sample.metadata["source_index"] == 0


def test_materialize_omnidocbench_sample_rejects_unknown_category(tmp_path: Path) -> None:
    raw_data_dir = tmp_path / "raw"
    create_fake_pdf_folders(raw_data_dir)

    raw_record = make_fake_raw_record()
    raw_record["layout_dets"][0]["category_type"] = "unknown_new_category"

    with pytest.raises(ValueError):
        materialize_omnidocbench_sample(
            raw_record=raw_record,
            source_index=0,
            raw_data_dir=raw_data_dir,
            output_dir=tmp_path / "out",
        )


def test_pdf_path_rejects_unknown_input_type(tmp_path: Path) -> None:
    raw_data_dir = tmp_path / "raw"
    create_fake_pdf_folders(raw_data_dir)

    sample = materialize_omnidocbench_sample(
        raw_record=make_fake_raw_record(),
        source_index=0,
        raw_data_dir=raw_data_dir,
        output_dir=tmp_path / "out",
    )

    with pytest.raises(ValueError):
        sample.pdf_path("bad_mode")


def test_to_manifest_record_is_json_serializable(tmp_path: Path) -> None:
    raw_data_dir = tmp_path / "raw"
    create_fake_pdf_folders(raw_data_dir)

    sample = materialize_omnidocbench_sample(
        raw_record=make_fake_raw_record(),
        source_index=0,
        raw_data_dir=raw_data_dir,
        output_dir=tmp_path / "out",
    )

    record = sample.to_manifest_record()

    serialized = json.dumps(record)
    round_tripped = json.loads(serialized)

    assert round_tripped["sample_id"] == sample.sample_id
    assert round_tripped["pdf_path_digital"] == str(sample.pdf_path_digital)
    assert round_tripped["pdf_path_scanned"] == str(sample.pdf_path_scanned)
    assert round_tripped["has_table"] is True
    assert round_tripped["has_formula"] is True


def test_save_and_load_manifest_with_limit(tmp_path: Path) -> None:
    raw_data_dir = tmp_path / "raw"
    create_fake_pdf_folders(raw_data_dir)

    sample_1 = materialize_omnidocbench_sample(
        raw_record=make_fake_raw_record(),
        source_index=0,
        raw_data_dir=raw_data_dir,
        output_dir=tmp_path / "out",
    )

    raw_record_2 = make_fake_raw_record()
    raw_record_2["page_info"]["image_path"] = "page-test-002.png"

    (raw_data_dir / "pdfs" / "page-test-002.pdf").write_bytes(FAKE_PDF_BYTES)
    (raw_data_dir / "ori_pdfs" / "page-test-002.pdf").write_bytes(FAKE_PDF_BYTES)

    sample_2 = materialize_omnidocbench_sample(
        raw_record=raw_record_2,
        source_index=1,
        raw_data_dir=raw_data_dir,
        output_dir=tmp_path / "out",
    )

    manifest_path = tmp_path / "manifest.jsonl"

    save_manifest(
        samples=[sample_1, sample_2],
        manifest_path=manifest_path,
    )

    loaded_samples = iter_omnidocbench_samples_from_manifest(
        manifest_path=manifest_path,
        limit=1,
    )

    assert len(loaded_samples) == 1
    assert loaded_samples[0].sample_id == sample_1.sample_id
    assert loaded_samples[0].image_filename == "page-test-001.png"
    assert loaded_samples[0].has_table is True


def test_sample_from_manifest_record_restores_paths_and_fields(tmp_path: Path) -> None:
    record = {
        "sample_id": "omnidocbench_000123_test",
        "source_index": 123,
        "image_filename": "page-test-123.png",
        "pdf_path_digital": str(tmp_path / "ori_pdfs" / "page-test-123.pdf"),
        "pdf_path_scanned": str(tmp_path / "pdfs" / "page-test-123.pdf"),
        "gt_layout_dets": [{"category_type": "text_block"}],
        "gt_text_blocks": [{"category_type": "text_block"}],
        "gt_tables": [],
        "gt_reading_order": [{"anno_id": "box_1", "order": 1}],
        "gt_bboxes_xyxy": [[1, 2, 3, 4]],
        "gt_bboxes_raw_poly": [[1, 2, 3, 2, 3, 4, 1, 4]],
        "page_width": 1000,
        "page_height": 2000,
        "data_source": "book",
        "language": "english",
        "layout": "single_column",
        "special_issues": ["watermark"],
        "has_table": False,
        "has_formula": False,
        "has_figure": False,
        "metadata": {"dataset_slug": "opendatalab/OmniDocBench"},
    }

    sample = sample_from_manifest_record(record)

    assert sample.sample_id == "omnidocbench_000123_test"
    assert sample.source_index == 123
    assert sample.image_filename == "page-test-123.png"
    assert sample.pdf_path_digital == tmp_path / "ori_pdfs" / "page-test-123.pdf"
    assert sample.pdf_path_scanned == tmp_path / "pdfs" / "page-test-123.pdf"
    assert sample.gt_bboxes_xyxy == [[1.0, 2.0, 3.0, 4.0]]
    assert sample.page_width == 1000.0
    assert sample.page_height == 2000.0
    assert sample.special_issues == ["watermark"]


def test_load_gt_json_reads_list(tmp_path: Path) -> None:
    raw_data_dir = tmp_path / "raw"
    raw_data_dir.mkdir(parents=True)

    gt_path = raw_data_dir / "OmniDocBench.json"
    gt_path.write_text(json.dumps([make_fake_raw_record()]), encoding="utf-8")

    data = load_gt_json(raw_data_dir)

    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["page_info"]["image_path"] == "page-test-001.png"


def test_prepare_omnidocbench_uses_existing_local_files(tmp_path: Path) -> None:
    raw_data_dir = tmp_path / "raw"
    create_fake_pdf_folders(raw_data_dir)

    gt_path = raw_data_dir / "OmniDocBench.json"
    gt_path.write_text(json.dumps([make_fake_raw_record()]), encoding="utf-8")

    manifest_path = prepare_omnidocbench(
        raw_data_dir=raw_data_dir,
        output_dir=tmp_path / "out",
        limit=1,
    )

    assert manifest_path.exists()

    samples = iter_omnidocbench_samples_from_manifest(
        manifest_path=manifest_path,
        limit=1,
    )

    assert len(samples) == 1
    assert samples[0].image_filename == "page-test-001.png"
    assert samples[0].pdf_path_digital.exists()
    assert samples[0].pdf_path_scanned.exists()