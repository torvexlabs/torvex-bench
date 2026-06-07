from __future__ import annotations

import json
from pathlib import Path

import pytest

from torvex_bench.harnesses import official_doclaynet as mod
from torvex_bench.harnesses.docling_eval_doclaynet import (
    DocLayNetPredictionSummary,
)


def test_stats_mean_reads_nested_mean() -> None:
    payload = {
        "map_50_stats": {
            "mean": 0.75,
        }
    }

    assert mod._stats_mean(payload, "map_50_stats") == 0.75
    assert mod._stats_mean(payload, "missing_stats") == -1.0


def test_official_result_path() -> None:
    work_dir = Path("benchmarks/docling_eval/DocLayNetV1_dev")

    assert mod._official_result_path(work_dir) == Path(
        "benchmarks/docling_eval/DocLayNetV1_dev"
        "/results/evaluations/layout/evaluation_DocLayNetV1_layout.json"
    )


def test_run_official_doclaynet_orchestrates_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work_dir = tmp_path / "DocLayNetV1_dev"
    commands: list[list[str]] = []

    def fake_run_command(command: list[str]) -> None:
        commands.append(command)

        if command[1] == "evaluate":
            result_path = mod._official_result_path(work_dir)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps(
                    {
                        "evaluated_samples": 3,
                        "mAP": 0.25,
                        "map_50_stats": {"mean": 0.50},
                        "map_75_stats": {"mean": 0.20},
                        "map_stats": {"mean": 0.25},
                        "weighted_map_50_stats": {"mean": 0.80},
                        "rejected_samples": {
                            "invalid_conversion_status": 0,
                            "mismatched_document": 0,
                            "missing_prediction": 0,
                        },
                        "true_labels": {"text": 3},
                        "pred_labels": {"text": 2},
                        "intersecting_labels": ["text"],
                        "evaluations_per_class": [
                            {
                                "label": "text",
                                "name": "Class AP[0.5:0.95]",
                                "value": 0.25,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

    def fake_generate_doclaynet_predictions(**kwargs):
        assert kwargs["gt_dir"] == work_dir / "gt_dataset" / "test"
        assert kwargs["prediction_dir"] == work_dir / "predictions" / "torvex_extract"
        assert kwargs["limit"] == 3
        assert kwargs["overwrite"] is True
        assert kwargs["save_normalized"] is True
        assert kwargs["device"] == "gpu"

        return DocLayNetPredictionSummary(
            requested=3,
            processed=3,
            predictions_written=3,
            skipped_existing=0,
            errors=0,
            prediction_dir=kwargs["prediction_dir"],
            normalized_dir=work_dir / "predictions" / "normalized",
        )

    monkeypatch.setattr(mod, "_run_command", fake_run_command)
    monkeypatch.setattr(
        mod,
        "generate_doclaynet_predictions",
        fake_generate_doclaynet_predictions,
    )

    summary = mod.run_official_doclaynet(
        limit=3,
        work_dir=work_dir,
        clean=True,
        save_normalized=True,
        device="gpu",
    )

    assert summary.limit == 3
    assert summary.device == "gpu"
    assert summary.evaluated_samples == 3
    assert summary.map_score == 0.25
    assert summary.map_50_mean == 0.50
    assert summary.weighted_map_50_mean == 0.80
    assert summary.prediction_summary.predictions_written == 3
    assert summary.summary_path.exists()

    compact = json.loads(summary.summary_path.read_text(encoding="utf-8"))
    assert compact["benchmark"] == "DocLayNetV1"
    assert compact["engine"] == "torvex_extract"
    assert compact["device"] == "gpu"
    assert compact["prediction_summary"]["errors"] == 0

    assert len(commands) == 3
    assert commands[0][1] == "create-gt"
    assert commands[1][1] == "create-eval"
    assert commands[2][1] == "evaluate"


def test_run_official_doclaynet_rejects_invalid_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="limit"):
        mod.run_official_doclaynet(
            limit=0,
            work_dir=tmp_path / "DocLayNetV1_dev",
        )


def test_run_official_doclaynet_rejects_invalid_device(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="device"):
        mod.run_official_doclaynet(
            limit=1,
            work_dir=tmp_path / "DocLayNetV1_dev",
            device="tpu",
        )