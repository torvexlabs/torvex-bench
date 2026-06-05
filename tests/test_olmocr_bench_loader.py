from __future__ import annotations

import json
from pathlib import Path

import pytest

from torvex_bench.datasets import olmocr_bench as m


def make_math_row(
    *,
    pdf: str = "arxiv_math/2503.04048_pg46.pdf",
    row_id: str = "math_000",
    math: str = r"{\mathcal{V}}(\psi_m)\rightarrow +\infty",
) -> dict:
    return {
        "pdf": pdf,
        "url": "https://arxiv.org/pdf/2503.04048",
        "page": 1,
        "id": row_id,
        "type": "math",
        "max_diffs": 0,
        "checked": None,
        "math": math,
    }


def make_text_row(
    *,
    pdf: str = "old_scans/1.pdf",
    row_id: str = "text_000",
    test_type: str = "present",
    text: str = "as we are deprived of Voting",
) -> dict:
    return {
        "pdf": pdf,
        "page": 1,
        "id": row_id,
        "type": test_type,
        "max_diffs": 1,
        "text": text,
        "case_sensitive": True,
        "first_n": None,
        "last_n": None,
        "checked": "verified",
        "url": "https://example.com/source",
    }


def make_order_row(
    *,
    pdf: str = "multi_column/example_pg1.pdf",
    row_id: str = "order_000",
) -> dict:
    return {
        "pdf": pdf,
        "page": 1,
        "id": row_id,
        "type": "order",
        "before": "first phrase",
        "after": "second phrase",
        "max_diffs": 2,
        "checked": "verified",
        "url": "https://example.com/order",
    }


def make_table_row(
    *,
    pdf: str = "tables/example_table.pdf",
    row_id: str = "table_000",
) -> dict:
    return {
        "pdf": pdf,
        "page": 1,
        "id": row_id,
        "type": "table",
        "max_diffs": 0,
        "checked": "verified",
        "cell": "0.569",
        "up": None,
        "down": None,
        "left": None,
        "right": None,
        "top_heading": None,
        "left_heading": "BO",
        "url": "https://example.com/table",
    }


def build_one_sample(tmp_path: Path) -> m.OlmOCRBenchSample:
    raw_rows = [
        ("old_scans", make_text_row(), 0),
        (
            "old_scans",
            make_text_row(row_id="text_001", text="another expected phrase"),
            1,
        ),
    ]
    grouped = m.group_tests_by_pdf(raw_rows)
    pdf_repo_path, grouped_record = next(iter(grouped.items()))

    return m.materialize_olmocr_sample(
        pdf_repo_path=pdf_repo_path,
        grouped_record=grouped_record,
        source_index=0,
        subset="non_math",
        raw_data_dir=tmp_path,
        download_pdfs=False,
    )


def test_categories_for_subset_and_expected_counts() -> None:
    assert set(m.categories_for_subset("non_math")) == {
        "headers_footers",
        "long_tiny_text",
        "multi_column",
        "old_scans",
        "table_tests",
    }

    assert set(m.categories_for_subset("math")) == {
        "arxiv_math",
        "old_scans_math",
    }

    assert set(m.categories_for_subset("all")) == set(m.TEST_FILES)

    assert m.expected_test_count_for_subset("non_math") == 3634
    assert m.expected_test_count_for_subset("math") == 3385
    assert m.expected_test_count_for_subset("all") == 7019

    with pytest.raises(ValueError):
        m.categories_for_subset("bad_subset")

    with pytest.raises(ValueError):
        m.expected_test_count_for_subset("bad_subset")


def test_make_repo_pdf_path() -> None:
    assert (
        m.make_repo_pdf_path("arxiv_math/2503.04048_pg46.pdf")
        == "bench_data/pdfs/arxiv_math/2503.04048_pg46.pdf"
    )

    assert (
        m.make_repo_pdf_path("tables/b5c5b866_pg4.pdf")
        == "bench_data/pdfs/tables/b5c5b866_pg4.pdf"
    )

    assert (
        m.make_repo_pdf_path("\\old_scans\\1.pdf")
        == "bench_data/pdfs/old_scans/1.pdf"
    )

    with pytest.raises(ValueError):
        m.make_repo_pdf_path("")


def test_normalize_math_test_case_payload() -> None:
    row = make_math_row()

    test_case = m.normalize_test_case(
        row=row,
        category="arxiv_math",
        source_file="bench_data/arxiv_math.jsonl",
        source_index=0,
    )

    assert test_case.test_id == "math_000"
    assert test_case.category == "arxiv_math"
    assert test_case.test_type == "math"
    assert test_case.page == 1
    assert test_case.max_diffs == 0
    assert test_case.checked is None
    assert test_case.payload == {
        "math": r"{\mathcal{V}}(\psi_m)\rightarrow +\infty"
    }
    assert test_case.metadata["source_file"] == "bench_data/arxiv_math.jsonl"
    assert test_case.metadata["source_index"] == 0
    assert test_case.metadata["url"] == "https://arxiv.org/pdf/2503.04048"


def test_normalize_text_test_case_payload() -> None:
    row = make_text_row(test_type="absent", text="page footer")

    test_case = m.normalize_test_case(
        row=row,
        category="old_scans",
        source_file="bench_data/old_scans.jsonl",
        source_index=5,
    )

    assert test_case.test_id == "text_000"
    assert test_case.category == "old_scans"
    assert test_case.test_type == "absent"
    assert test_case.checked == "verified"
    assert test_case.payload == {
        "text": "page footer",
        "case_sensitive": True,
        "first_n": None,
        "last_n": None,
    }


def test_normalize_order_and_table_payloads() -> None:
    order_case = m.normalize_test_case(
        row=make_order_row(),
        category="multi_column",
        source_file="bench_data/multi_column.jsonl",
        source_index=0,
    )

    assert order_case.test_type == "order"
    assert order_case.payload == {
        "before": "first phrase",
        "after": "second phrase",
    }

    table_case = m.normalize_test_case(
        row=make_table_row(),
        category="table_tests",
        source_file="bench_data/table_tests.jsonl",
        source_index=0,
    )

    assert table_case.test_type == "table"
    assert table_case.payload == {
        "cell": "0.569",
        "up": None,
        "down": None,
        "left": None,
        "right": None,
        "top_heading": None,
        "left_heading": "BO",
    }


def test_normalize_test_case_requires_pdf_and_type() -> None:
    row_missing_pdf = make_text_row()
    row_missing_pdf.pop("pdf")

    with pytest.raises(ValueError, match="missing pdf"):
        m.normalize_test_case(
            row=row_missing_pdf,
            category="old_scans",
            source_file="bench_data/old_scans.jsonl",
            source_index=0,
        )

    row_missing_type = make_text_row()
    row_missing_type.pop("type")

    with pytest.raises(ValueError, match="missing type"):
        m.normalize_test_case(
            row=row_missing_type,
            category="old_scans",
            source_file="bench_data/old_scans.jsonl",
            source_index=0,
        )


def test_group_tests_by_pdf_groups_multiple_tests_for_same_pdf() -> None:
    raw_rows = [
        ("old_scans", make_text_row(row_id="text_000"), 0),
        ("old_scans", make_text_row(row_id="text_001", text="second check"), 1),
        ("multi_column", make_order_row(), 0),
    ]

    grouped = m.group_tests_by_pdf(raw_rows)

    assert set(grouped) == {
        "bench_data/pdfs/old_scans/1.pdf",
        "bench_data/pdfs/multi_column/example_pg1.pdf",
    }

    old_scan_group = grouped["bench_data/pdfs/old_scans/1.pdf"]
    assert old_scan_group["raw_pdf_name"] == "old_scans/1.pdf"
    assert old_scan_group["categories"] == {"old_scans"}
    assert len(old_scan_group["tests"]) == 2
    assert [test.test_id for test in old_scan_group["tests"]] == [
        "text_000",
        "text_001",
    ]


def test_materialize_sample_without_downloading(tmp_path: Path) -> None:
    sample = build_one_sample(tmp_path)

    assert sample.sample_id.startswith("olmocr_")
    assert sample.source_index == 0
    assert sample.subset == "non_math"
    assert sample.pdf_repo_path == "bench_data/pdfs/old_scans/1.pdf"
    assert sample.raw_pdf_name == "old_scans/1.pdf"
    assert sample.pdf_path == tmp_path / "bench_data/pdfs/old_scans/1.pdf"
    assert sample.categories == ["old_scans"]
    assert sample.has_math is False
    assert sample.test_count == 2
    assert len(sample.tests) == 2
    assert sample.metadata["score_metric"] == "unit_test_pass_rate"


def test_save_and_load_manifest_roundtrip(tmp_path: Path) -> None:
    sample = build_one_sample(tmp_path)
    manifest_path = tmp_path / "manifest.jsonl"

    m.save_manifest([sample], manifest_path)

    records = m.load_manifest(manifest_path)
    assert len(records) == 1
    assert records[0]["sample_id"] == sample.sample_id
    assert records[0]["rank"] == 0
    assert records[0]["test_count"] == 2

    loaded_samples = m.iter_olmocr_samples_from_manifest(manifest_path)
    assert len(loaded_samples) == 1

    loaded = loaded_samples[0]
    assert loaded.sample_id == sample.sample_id
    assert loaded.pdf_path == sample.pdf_path
    assert loaded.pdf_repo_path == sample.pdf_repo_path
    assert loaded.categories == ["old_scans"]
    assert loaded.test_count == 2
    assert [test.test_id for test in loaded.tests] == ["text_000", "text_001"]


def test_manifest_counts(tmp_path: Path) -> None:
    sample = build_one_sample(tmp_path)
    manifest_path = tmp_path / "manifest.jsonl"

    assert m._manifest_counts(manifest_path) == (0, 0)

    m.save_manifest([sample], manifest_path)

    assert m._manifest_counts(manifest_path) == (1, 2)


def test_manifest_sufficiency_requires_existing_pdfs(tmp_path: Path) -> None:
    sample = build_one_sample(tmp_path)
    manifest_path = tmp_path / "manifest.jsonl"

    m.save_manifest([sample], manifest_path)

    assert m._manifest_pdfs_exist(manifest_path) is False

    assert (
        m._manifest_is_sufficient(
            manifest_path,
            limit=1,
            subset="non_math",
            require_pdfs=True,
        )
        is False
    )

    assert (
        m._manifest_is_sufficient(
            manifest_path,
            limit=1,
            subset="non_math",
            require_pdfs=False,
        )
        is True
    )

    sample.pdf_path.parent.mkdir(parents=True, exist_ok=True)
    sample.pdf_path.write_bytes(b"%PDF-1.4 fake test pdf\n")

    assert m._manifest_pdfs_exist(manifest_path) is True

    assert (
        m._manifest_is_sufficient(
            manifest_path,
            limit=1,
            subset="non_math",
            require_pdfs=True,
        )
        is True
    )


def test_full_manifest_sufficiency_uses_test_count_not_sample_count(
    tmp_path: Path,
) -> None:
    sample = build_one_sample(tmp_path)
    sample.pdf_path.parent.mkdir(parents=True, exist_ok=True)
    sample.pdf_path.write_bytes(b"%PDF-1.4 fake test pdf\n")

    manifest_path = tmp_path / "manifest.jsonl"
    m.save_manifest([sample], manifest_path)

    assert sample.test_count == 2

    assert (
        m._manifest_is_sufficient(
            manifest_path,
            limit=None,
            subset="non_math",
            require_pdfs=True,
        )
        is False
    )


def test_materialize_dataset_applies_limit_after_grouping_without_pdf_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_rows = [
        ("old_scans", make_text_row(pdf="old_scans/1.pdf", row_id="a"), 0),
        ("old_scans", make_text_row(pdf="old_scans/2.pdf", row_id="b"), 1),
        ("old_scans", make_text_row(pdf="old_scans/3.pdf", row_id="c"), 2),
    ]

    monkeypatch.setattr(
        m,
        "load_olmocr_test_rows",
        lambda raw_data_dir, *, subset: fake_rows,
    )

    manifest_path = tmp_path / "out" / "manifest.jsonl"

    samples = m.materialize_olmocr_bench_dataset(
        raw_data_dir=tmp_path / "raw",
        subset="non_math",
        output_dir=tmp_path / "out",
        limit=1,
        manifest_path=manifest_path,
        download_pdfs=False,
    )

    assert len(samples) == 1
    assert samples[0].raw_pdf_name == "old_scans/1.pdf"

    records = m.load_manifest(manifest_path)
    assert len(records) == 1
    assert records[0]["raw_pdf_name"] == "old_scans/1.pdf"


def test_prepare_returns_existing_sufficient_manifest_without_rematerializing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = build_one_sample(tmp_path)
    sample.pdf_path.parent.mkdir(parents=True, exist_ok=True)
    sample.pdf_path.write_bytes(b"%PDF-1.4 fake test pdf\n")

    manifest_path = tmp_path / "manifest.jsonl"
    m.save_manifest([sample], manifest_path)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("materialize_olmocr_bench_dataset should not be called")

    monkeypatch.setattr(m, "materialize_olmocr_bench_dataset", fail_if_called)

    result = m.prepare_olmocr_bench(
        raw_data_dir=tmp_path,
        subset="non_math",
        output_dir=tmp_path / "out",
        limit=1,
        manifest_path=manifest_path,
        download_pdfs=True,
    )

    assert result == manifest_path


def test_prepare_rematerializes_when_manifest_pdf_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = build_one_sample(tmp_path)
    manifest_path = tmp_path / "manifest.jsonl"
    m.save_manifest([sample], manifest_path)

    calls = {"count": 0}

    def fake_materialize_olmocr_bench_dataset(*args, **kwargs):
        calls["count"] += 1
        sample.pdf_path.parent.mkdir(parents=True, exist_ok=True)
        sample.pdf_path.write_bytes(b"%PDF-1.4 fake test pdf\n")
        m.save_manifest([sample], kwargs["manifest_path"])
        return [sample]

    monkeypatch.setattr(
        m,
        "materialize_olmocr_bench_dataset",
        fake_materialize_olmocr_bench_dataset,
    )

    result = m.prepare_olmocr_bench(
        raw_data_dir=tmp_path,
        subset="non_math",
        output_dir=tmp_path / "out",
        limit=1,
        manifest_path=manifest_path,
        download_pdfs=True,
    )

    assert result == manifest_path
    assert calls["count"] == 1


def test_get_hf_dataset_commit_never_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_dataset_info(slug: str):
        raise RuntimeError("network unavailable")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "dataset_info", fail_dataset_info)

    assert m.get_hf_dataset_commit() == "unknown"