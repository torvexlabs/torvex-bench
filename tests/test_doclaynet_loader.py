import base64
import json
from pathlib import Path

from torvex_bench.datasets.doclaynet import (
    DocLayNetSample,
    coerce_pdf_bytes,
    extract_page_pdf,
    iter_doclaynet_samples_from_manifest,
    make_sample_id,
    materialize_doclaynet_sample,
    normalize_doclaynet_bbox,
    save_manifest,
    sample_from_manifest_record,
)


FAKE_PDF_BYTES = b"%PDF-1.4\n% fake test pdf\n%%EOF"


def test_make_sample_id_is_stable() -> None:
    sample_id_1 = make_sample_id(
        split="test",
        source_index=7,
        category_ids=[9, 10, 1],
        bbox_count=3,
    )

    sample_id_2 = make_sample_id(
        split="test",
        source_index=7,
        category_ids=[9, 10, 1],
        bbox_count=3,
    )

    assert sample_id_1 == sample_id_2
    assert sample_id_1.startswith("doclaynet_test_000007_")


def test_coerce_pdf_bytes_accepts_bytes() -> None:
    assert coerce_pdf_bytes(FAKE_PDF_BYTES) == FAKE_PDF_BYTES


def test_coerce_pdf_bytes_accepts_base64_string() -> None:
    encoded = base64.b64encode(FAKE_PDF_BYTES).decode("utf-8")

    assert coerce_pdf_bytes(encoded) == FAKE_PDF_BYTES


def test_extract_page_pdf_writes_pdf_bytes(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"

    extract_page_pdf(
        pdf_bytes=FAKE_PDF_BYTES,
        pdf_path=pdf_path,
    )

    assert pdf_path.exists()
    assert pdf_path.read_bytes() == FAKE_PDF_BYTES


def test_normalize_doclaynet_bbox_converts_coco_xywh_to_xyxy() -> None:
    result = normalize_doclaynet_bbox([10, 20, 30, 40])

    assert result == [10.0, 20.0, 40.0, 60.0]


def test_materialize_doclaynet_sample_creates_pdf_and_sample(tmp_path: Path) -> None:
    raw_sample = {
        "pdf": FAKE_PDF_BYTES,
        "bboxes": [
            [10, 20, 30, 40],
            [100, 200, 50, 60],
        ],
        "category_id": [9, 10],
        "metadata": {
            "coco_width": 612,
            "coco_height": 792,
            "doc_category": "financial_reports",
        },
        "image": None,
        "segmentation": [],
        "area": [],
        "pdf_cells": [],
        "modalities": [],
    }

    sample = materialize_doclaynet_sample(
        raw_sample=raw_sample,
        source_index=0,
        output_dir=tmp_path,
    )

    assert isinstance(sample, DocLayNetSample)
    assert sample.source_index == 0
    assert sample.split == "test"

    assert sample.pdf_path.exists()
    assert sample.pdf_path.suffix == ".pdf"

    assert sample.gt_bboxes_raw == [
        [10.0, 20.0, 30.0, 40.0],
        [100.0, 200.0, 50.0, 60.0],
    ]

    assert sample.gt_bboxes == [
        [10.0, 20.0, 40.0, 60.0],
        [100.0, 200.0, 150.0, 260.0],
    ]

    assert sample.gt_category_ids == [9, 10]
    assert sample.gt_categories == ["Table", "Text"]

    assert sample.page_width == 612.0
    assert sample.page_height == 792.0

    assert sample.has_table is True
    assert sample.has_formula is False

    assert sample.metadata["dataset_slug"] == "docling-project/DocLayNet-v1.2"
    assert sample.metadata["bbox_input_format"] == "coco_xywh"
    assert sample.metadata["bbox_output_format"] == "xyxy"


def test_to_manifest_record_is_json_serializable(tmp_path: Path) -> None:
    sample = DocLayNetSample(
        sample_id="sample_1",
        source_index=0,
        split="test",
        pdf_path=tmp_path / "sample_1.pdf",
        gt_bboxes=[[10.0, 20.0, 40.0, 60.0]],
        gt_bboxes_raw=[[10.0, 20.0, 30.0, 40.0]],
        gt_category_ids=[9],
        gt_categories=["Table"],
        page_width=612.0,
        page_height=792.0,
        bbox_format="xyxy_from_coco_xywh",
        has_table=True,
        has_formula=False,
        metadata={"dataset_slug": "docling-project/DocLayNet-v1.2"},
    )

    record = sample.to_manifest_record()

    serialized = json.dumps(record)
    round_tripped = json.loads(serialized)

    assert round_tripped["pdf_path"] == str(tmp_path / "sample_1.pdf")
    assert round_tripped["gt_category_ids"] == [9]
    assert round_tripped["gt_categories"] == ["Table"]


def test_save_and_load_manifest_with_limit(tmp_path: Path) -> None:
    sample_1 = DocLayNetSample(
        sample_id="sample_1",
        source_index=0,
        split="test",
        pdf_path=tmp_path / "sample_1.pdf",
        gt_bboxes=[[10.0, 20.0, 40.0, 60.0]],
        gt_bboxes_raw=[[10.0, 20.0, 30.0, 40.0]],
        gt_category_ids=[9],
        gt_categories=["Table"],
        page_width=612.0,
        page_height=792.0,
        bbox_format="xyxy_from_coco_xywh",
        has_table=True,
        has_formula=False,
        metadata={"dataset_slug": "docling-project/DocLayNet-v1.2"},
    )

    sample_2 = DocLayNetSample(
        sample_id="sample_2",
        source_index=1,
        split="test",
        pdf_path=tmp_path / "sample_2.pdf",
        gt_bboxes=[[100.0, 200.0, 150.0, 260.0]],
        gt_bboxes_raw=[[100.0, 200.0, 50.0, 60.0]],
        gt_category_ids=[10],
        gt_categories=["Text"],
        page_width=612.0,
        page_height=792.0,
        bbox_format="xyxy_from_coco_xywh",
        has_table=False,
        has_formula=False,
        metadata={"dataset_slug": "docling-project/DocLayNet-v1.2"},
    )

    manifest_path = tmp_path / "manifest.jsonl"

    save_manifest(
        samples=[sample_1, sample_2],
        manifest_path=manifest_path,
    )

    loaded_samples = iter_doclaynet_samples_from_manifest(
        manifest_path=manifest_path,
        limit=1,
    )

    assert len(loaded_samples) == 1
    assert loaded_samples[0].sample_id == "sample_1"
    assert loaded_samples[0].gt_category_ids == [9]
    assert loaded_samples[0].gt_categories == ["Table"]


def test_sample_from_manifest_record_restores_paths_and_fields(tmp_path: Path) -> None:
    record = {
        "sample_id": "sample_123",
        "source_index": 123,
        "split": "test",
        "pdf_path": str(tmp_path / "sample_123.pdf"),
        "gt_bboxes": [[10.0, 20.0, 40.0, 60.0]],
        "gt_bboxes_raw": [[10.0, 20.0, 30.0, 40.0]],
        "gt_category_ids": [3],
        "gt_categories": ["Formula"],
        "page_width": 612.0,
        "page_height": 792.0,
        "bbox_format": "xyxy_from_coco_xywh",
        "has_table": False,
        "has_formula": True,
        "metadata": {"dataset_slug": "docling-project/DocLayNet-v1.2"},
    }

    sample = sample_from_manifest_record(record)

    assert sample.sample_id == "sample_123"
    assert sample.source_index == 123
    assert sample.pdf_path == tmp_path / "sample_123.pdf"
    assert sample.gt_category_ids == [3]
    assert sample.gt_categories == ["Formula"]
    assert sample.has_formula is True