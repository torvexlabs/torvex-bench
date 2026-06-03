from pathlib import Path

from PIL import Image

from torvex_bench.datasets.fintabnet import (
    FinTabNetSample,
    detect_spans,
    iter_fintabnet_samples_from_manifest,
    make_sample_id,
    materialize_fintabnet_sample,
    normalize_html,
    normalize_otsl,
    save_manifest,
    sample_from_manifest_record,
)


def test_normalize_html_joins_html_token_list() -> None:
    html_tokens = ["<tr>", "<td>", "hello", "</td>", "</tr>"]

    assert normalize_html(html_tokens) == "<tr><td>hello</td></tr>"


def test_normalize_html_handles_none() -> None:
    assert normalize_html(None) == ""


def test_normalize_otsl_joins_token_list() -> None:
    otsl_tokens = ["fcel", "fcel", "nl", "lcel"]

    assert normalize_otsl(otsl_tokens) == "fcel fcel nl lcel"


def test_detect_spans_detects_merged_cell_tokens() -> None:
    assert detect_spans("fcel fcel nl fcel lcel nl") is True
    assert detect_spans("fcel fcel nl fcel ucel nl") is True
    assert detect_spans("fcel fcel nl fcel xcel nl") is True
    assert detect_spans("fcel fcel nl fcel ecel nl") is False


def test_make_sample_id_is_stable() -> None:
    sample_id_1 = make_sample_id(
        split="test",
        source_index=7,
        rows=3,
        cols=4,
        gt_otsl="fcel fcel nl",
        gt_html="<tr><td></td></tr>",
    )

    sample_id_2 = make_sample_id(
        split="test",
        source_index=7,
        rows=3,
        cols=4,
        gt_otsl="fcel fcel nl",
        gt_html="<tr><td></td></tr>",
    )

    assert sample_id_1 == sample_id_2
    assert sample_id_1.startswith("fintabnet_test_000007_")


def test_materialize_fintabnet_sample_creates_png_pdf_and_sample(tmp_path: Path) -> None:
    image = Image.new("RGB", (120, 80), "white")

    raw_sample = {
        "rows": 2,
        "cols": 2,
        "otsl": ["fcel", "fcel", "nl", "fcel", "lcel", "nl"],
        "html": ["<tr>", "<td>", "A", "</td>", "<td>", "B", "</td>", "</tr>"],
        "html_restored": ["<tr>", "<td>", "A", "</td>", "<td>", "B", "</td>", "</tr>"],
        "image": image,
    }

    sample = materialize_fintabnet_sample(
        raw_sample=raw_sample,
        source_index=0,
        output_dir=tmp_path,
    )

    assert isinstance(sample, FinTabNetSample)
    assert sample.source_index == 0
    assert sample.rows == 2
    assert sample.cols == 2
    assert sample.has_spans is True

    assert sample.gt_html == "<tr><td>A</td><td>B</td></tr>"
    assert sample.image_path.exists()
    assert sample.pdf_path.exists()

    assert sample.image_path.suffix == ".png"
    assert sample.pdf_path.suffix == ".pdf"


def test_save_and_load_manifest_with_limit(tmp_path: Path) -> None:
    sample_1 = FinTabNetSample(
        sample_id="sample_1",
        source_index=0,
        split="test",
        pdf_path=tmp_path / "sample_1.pdf",
        image_path=tmp_path / "sample_1.png",
        gt_html="<tr><td>A</td></tr>",
        gt_html_restored="<tr><td>A</td></tr>",
        gt_otsl="fcel nl",
        rows=1,
        cols=1,
        has_spans=False,
        metadata={"dataset_slug": "docling-project/FinTabNet_OTSL"},
    )

    sample_2 = FinTabNetSample(
        sample_id="sample_2",
        source_index=1,
        split="test",
        pdf_path=tmp_path / "sample_2.pdf",
        image_path=tmp_path / "sample_2.png",
        gt_html="<tr><td>B</td></tr>",
        gt_html_restored="<tr><td>B</td></tr>",
        gt_otsl="fcel nl",
        rows=1,
        cols=1,
        has_spans=False,
        metadata={"dataset_slug": "docling-project/FinTabNet_OTSL"},
    )

    manifest_path = tmp_path / "manifest.jsonl"

    save_manifest(
        samples=[sample_1, sample_2],
        manifest_path=manifest_path,
    )

    loaded_samples = iter_fintabnet_samples_from_manifest(
        manifest_path=manifest_path,
        limit=1,
    )

    assert len(loaded_samples) == 1
    assert loaded_samples[0].sample_id == "sample_1"
    assert loaded_samples[0].gt_html == "<tr><td>A</td></tr>"


def test_sample_from_manifest_record_restores_paths_and_fields(tmp_path: Path) -> None:
    record = {
        "sample_id": "sample_123",
        "source_index": 123,
        "split": "test",
        "pdf_path": str(tmp_path / "sample_123.pdf"),
        "image_path": str(tmp_path / "sample_123.png"),
        "gt_html": "<tr><td>A</td></tr>",
        "gt_html_restored": "<tr><td>A</td></tr>",
        "gt_otsl": "fcel nl",
        "rows": 1,
        "cols": 1,
        "has_spans": False,
        "metadata": {"dataset_slug": "docling-project/FinTabNet_OTSL"},
    }

    sample = sample_from_manifest_record(record)

    assert sample.sample_id == "sample_123"
    assert sample.source_index == 123
    assert sample.pdf_path == tmp_path / "sample_123.pdf"
    assert sample.image_path == tmp_path / "sample_123.png"
    assert sample.gt_html == "<tr><td>A</td></tr>"