import json
from pathlib import Path

import pytest

from torvex_bench.datasets.omnidocbench import (
    DATASET_SLUG,
    DEFAULT_INPUT_TYPE,
    EXPECTED_COUNT,
    GT_JSON_FILENAME,
    IMAGE_SUBDIR,
    INPUT_TYPE_SCANNED,
    OmniDocBenchSample,
    default_manifest_path,
    image_filename_from_page_info,
    image_repo_path,
    iter_omnidocbench_samples_from_manifest,
    local_image_path,
    make_sample_id,
    materialize_omnidocbench_dataset,
    materialize_omnidocbench_sample,
    poly_to_xyxy,
    prepare_omnidocbench,
    save_manifest,
)


FAKE_IMAGE_BYTES = b"\x89PNG\r\n\x1a\nfake omnidocbench image"


def make_raw_record(
    *,
    image_path: str = "page-d1561665-5359-42fe-920c-d6e3bff81953.png",
) -> dict:
    return {
        "page_info": {
            "image_path": image_path,
            "page_no": 1,
            "width": 1000,
            "height": 1400,
            "page_attribute": {
                "language": "english",
                "layout": "single_column",
            },
        },
        "layout_dets": [
            {
                "category_type": "title",
                "poly": [10, 20, 110, 20, 110, 60, 10, 60],
                "ignore": False,
                "order": 1,
                "anno_id": 1,
                "text": "Example title",
            },
            {
                "category_type": "text_block",
                "poly": [10, 80, 300, 80, 300, 140, 10, 140],
                "ignore": False,
                "order": 2,
                "anno_id": 2,
                "text": "Example paragraph",
            },
        ],
        "extra": {
            "source": "unit-test",
        },
    }


def write_fake_raw_dataset(
    raw_data_dir: Path,
    *,
    records: list[dict] | None = None,
) -> list[dict]:
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    if records is None:
        records = [make_raw_record()]

    (raw_data_dir / GT_JSON_FILENAME).write_text(
        json.dumps(records, ensure_ascii=False),
        encoding="utf-8",
    )

    for record in records:
        image_name = image_filename_from_page_info(record["page_info"])
        image_path = raw_data_dir / IMAGE_SUBDIR / image_name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(FAKE_IMAGE_BYTES)

    return records


def test_constants_are_scanned_image_only() -> None:
    assert INPUT_TYPE_SCANNED == "scanned"
    assert DEFAULT_INPUT_TYPE == "scanned"
    assert DATASET_SLUG == "opendatalab/OmniDocBench"
    assert EXPECTED_COUNT == 1651


def test_image_repo_path_prepends_images_folder() -> None:
    assert image_repo_path("page-abc.png") == "images/page-abc.png"
    assert image_repo_path("images/page-abc.png") == "images/page-abc.png"


def test_image_filename_from_page_info_normalizes_paths() -> None:
    assert (
        image_filename_from_page_info({"image_path": "page-abc.png"})
        == "page-abc.png"
    )
    assert (
        image_filename_from_page_info({"image_path": "images/page-abc.png"})
        == "page-abc.png"
    )
    assert (
        image_filename_from_page_info({"image_path": r"images\page-abc.png"})
        == "page-abc.png"
    )


def test_image_filename_from_page_info_rejects_missing_path() -> None:
    with pytest.raises(ValueError, match="page_info.image_path"):
        image_filename_from_page_info({})


def test_local_image_path_uses_images_subdir(tmp_path: Path) -> None:
    path = local_image_path("page-abc.png", raw_data_dir=tmp_path)

    assert path == tmp_path / "images" / "page-abc.png"


def test_make_sample_id_is_stable() -> None:
    sample_id_1 = make_sample_id(
        source_index=7,
        image_filename="page-abc.png",
    )
    sample_id_2 = make_sample_id(
        source_index=7,
        image_filename="page-abc.png",
    )

    assert sample_id_1 == sample_id_2
    assert sample_id_1.startswith("omnidocbench_000007_")


def test_poly_to_xyxy_converts_quadrilateral() -> None:
    assert poly_to_xyxy([10, 20, 110, 20, 110, 60, 10, 60]) == [
        10.0,
        20.0,
        110.0,
        60.0,
    ]


def test_poly_to_xyxy_rejects_bad_poly() -> None:
    with pytest.raises(ValueError, match="must have 8 floats"):
        poly_to_xyxy([1, 2, 3])


def test_materialize_omnidocbench_sample_builds_image_sample(tmp_path: Path) -> None:
    raw_data_dir = tmp_path / "raw"
    records = write_fake_raw_dataset(raw_data_dir)
    sample = materialize_omnidocbench_sample(
        raw_record=records[0],
        source_index=0,
        raw_data_dir=raw_data_dir,
    )

    assert isinstance(sample, OmniDocBenchSample)
    assert sample.source_index == 0
    assert sample.image_filename == "page-d1561665-5359-42fe-920c-d6e3bff81953.png"
    assert sample.image_repo_path == (
        "images/page-d1561665-5359-42fe-920c-d6e3bff81953.png"
    )
    assert sample.image_path.exists()
    assert sample.image_stem == "page-d1561665-5359-42fe-920c-d6e3bff81953"
    assert sample.prediction_filename == (
        "page-d1561665-5359-42fe-920c-d6e3bff81953.md"
    )
    assert sample.metadata["dataset_slug"] == DATASET_SLUG
    assert sample.metadata["prediction_filename"] == sample.prediction_filename


def test_materialize_omnidocbench_sample_rejects_missing_image(
    tmp_path: Path,
) -> None:
    raw_data_dir = tmp_path / "raw"
    raw_data_dir.mkdir(parents=True)
    raw_record = make_raw_record()

    with pytest.raises(FileNotFoundError, match="Missing OmniDocBench image"):
        materialize_omnidocbench_sample(
            raw_record=raw_record,
            source_index=0,
            raw_data_dir=raw_data_dir,
            require_image_exists=True,
        )


def test_manifest_roundtrip(tmp_path: Path) -> None:
    raw_data_dir = tmp_path / "raw"
    records = write_fake_raw_dataset(raw_data_dir)

    sample = materialize_omnidocbench_sample(
        raw_record=records[0],
        source_index=0,
        raw_data_dir=raw_data_dir,
    )

    manifest_path = tmp_path / "sample_manifest.jsonl"
    save_manifest([sample], manifest_path)

    loaded = iter_omnidocbench_samples_from_manifest(manifest_path)

    assert len(loaded) == 1
    assert loaded[0].sample_id == sample.sample_id
    assert loaded[0].image_filename == sample.image_filename
    assert loaded[0].prediction_filename == sample.prediction_filename


def test_iter_manifest_respects_limit(tmp_path: Path) -> None:
    raw_data_dir = tmp_path / "raw"
    records = [
        make_raw_record(image_path="page-a.png"),
        make_raw_record(image_path="page-b.png"),
    ]
    write_fake_raw_dataset(raw_data_dir, records=records)

    samples = [
        materialize_omnidocbench_sample(
            raw_record=record,
            source_index=index,
            raw_data_dir=raw_data_dir,
        )
        for index, record in enumerate(records)
    ]

    manifest_path = tmp_path / "sample_manifest.jsonl"
    save_manifest(samples, manifest_path)

    loaded = iter_omnidocbench_samples_from_manifest(manifest_path, limit=1)

    assert len(loaded) == 1
    assert loaded[0].image_filename == "page-a.png"


def test_default_manifest_path_uses_sample_manifest_name(tmp_path: Path) -> None:
    assert default_manifest_path(output_dir=tmp_path) == (
        tmp_path / "sample_manifest.jsonl"
    )


def test_materialize_omnidocbench_dataset_uses_existing_local_files(
    tmp_path: Path,
) -> None:
    raw_data_dir = tmp_path / "raw"
    output_dir = tmp_path / "prepared"
    write_fake_raw_dataset(raw_data_dir)

    samples = materialize_omnidocbench_dataset(
        raw_data_dir=raw_data_dir,
        output_dir=output_dir,
        limit=1,
        download_images=False,
    )

    assert len(samples) == 1
    assert samples[0].image_path.exists()
    assert (output_dir / "sample_manifest.jsonl").exists()


def test_prepare_omnidocbench_uses_existing_local_files(tmp_path: Path) -> None:
    raw_data_dir = tmp_path / "raw"
    output_dir = tmp_path / "prepared"
    write_fake_raw_dataset(raw_data_dir)

    manifest_path = prepare_omnidocbench(
        raw_data_dir=raw_data_dir,
        output_dir=output_dir,
        limit=1,
        download_images=False,
    )

    assert manifest_path == output_dir / "sample_manifest.jsonl"
    assert manifest_path.exists()

    samples = iter_omnidocbench_samples_from_manifest(manifest_path)
    assert len(samples) == 1
    assert samples[0].prediction_filename.endswith(".md")