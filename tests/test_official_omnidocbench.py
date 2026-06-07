import json
from pathlib import Path

from torvex_bench.datasets.omnidocbench import OmniDocBenchSample
from torvex_bench.harnesses.official_omnidocbench import (
    read_official_metrics,
    resolve_omnidocbench_eval_bin,
    write_official_omnidocbench_config,
    write_subset_gt_json,
)


def make_sample() -> OmniDocBenchSample:
    return OmniDocBenchSample(
        source_index=0,
        sample_id="omnidocbench_000000_test",
        image_filename="page-test.png",
        image_repo_path="images/page-test.png",
        image_path=Path("images/page-test.png"),
        page_info={
            "image_path": "page-test.png",
            "width": 100,
            "height": 200,
        },
        layout_dets=[
            {
                "category_type": "text_block",
                "text": "Hello",
                "ignore": False,
                "order": 1,
            }
        ],
        extra={"source": "unit-test"},
        metadata={},
    )


def test_write_subset_gt_json_writes_selected_samples(tmp_path: Path) -> None:
    output_path = tmp_path / "subset.json"

    result_path = write_subset_gt_json([make_sample()], output_path)

    assert result_path == output_path

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["page_info"]["image_path"] == "page-test.png"
    assert data[0]["layout_dets"][0]["text"] == "Hello"
    assert data[0]["extra"]["source"] == "unit-test"


def test_write_official_omnidocbench_config_omits_formula_cdm(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    gt_path = tmp_path / "gt.json"
    predictions_dir = tmp_path / "predictions" / "torvex_extract"

    write_official_omnidocbench_config(
        config_path=config_path,
        gt_json_path=gt_path,
        predictions_dir=predictions_dir,
        match_workers=1,
        teds_workers=1,
    )

    text = config_path.read_text(encoding="utf-8")

    assert "end2end_eval:" in text
    assert "dataset_name: end2end_dataset" in text
    assert "text_block:" in text
    assert "Edit_dist" in text
    assert "table:" in text
    assert "TEDS" in text
    assert "reading_order:" in text

    assert "display_formula" not in text
    assert "CDM" not in text
    assert "COCODet" not in text


def test_read_official_metrics_returns_empty_when_missing(tmp_path: Path) -> None:
    assert read_official_metrics(tmp_path / "missing.json") == {}


def test_read_official_metrics_reads_json(tmp_path: Path) -> None:
    path = tmp_path / "metric_result.json"
    path.write_text(
        json.dumps({"text_block": {"all": {"Edit_dist": {"ALL": 0.5}}}}),
        encoding="utf-8",
    )

    assert read_official_metrics(path)["text_block"]["all"]["Edit_dist"]["ALL"] == 0.5


def test_resolve_omnidocbench_eval_bin_accepts_explicit_path(tmp_path: Path) -> None:
    explicit = tmp_path / "omnidocbench-eval"
    assert resolve_omnidocbench_eval_bin(explicit) == explicit