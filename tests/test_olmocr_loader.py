from __future__ import annotations

import json
from pathlib import Path

import pytest

from torvex_bench.datasets import olmocr as m


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


def make_math_row(
    *,
    pdf: str = "arxiv_math/2503.04048_pg46.pdf",
    row_id: str = "math_000",
) -> dict:
    return {
        "pdf": pdf,
        "page": 1,
        "id": row_id,
        "type": "math",
        "max_diffs": 0,
        "checked": None,
        "math": r"{\mathcal{V}}(\psi_m)\rightarrow +\infty",
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_jsonl_files_for_track_and_expected_counts() -> None:
    assert m.jsonl_files_for_track("non_math") == [
        "headers_footers.jsonl",
        "long_tiny_text.jsonl",
        "multi_column.jsonl",
        "old_scans.jsonl",
        "table_tests.jsonl",
    ]

    assert m.jsonl_files_for_track("math") == [
        "arxiv_math.jsonl",
        "old_scans_math.jsonl",
    ]

    assert m.jsonl_files_for_track("full") == [
        "arxiv_math.jsonl",
        "headers_footers.jsonl",
        "long_tiny_text.jsonl",
        "multi_column.jsonl",
        "old_scans.jsonl",
        "old_scans_math.jsonl",
        "table_tests.jsonl",
    ]

    assert m.EXPECTED_TEST_COUNT == 7019
    assert m.EXPECTED_NON_MATH_TEST_COUNT == 3634
    assert m.EXPECTED_MATH_TEST_COUNT == 3385

    with pytest.raises(ValueError, match="track"):
        m.jsonl_files_for_track("bad")


def test_bench_data_and_manifest_paths(tmp_path: Path) -> None:
    work_dir = tmp_path / "olmocr"

    assert m.bench_data_dir(work_dir) == work_dir / "bench_data"

    # Important: manifest must stay outside bench_data.
    # Official olmOCR evaluator scans every *.jsonl inside bench_data.
    assert m.default_manifest_path(work_dir) == work_dir / "sample_manifest.jsonl"


def test_repo_and_local_paths(tmp_path: Path) -> None:
    work_dir = tmp_path / "olmocr"

    assert m.local_pdf_alias("old_scans/1.pdf") == "old_scans__1.pdf"
    assert m.local_pdf_alias(r"\old_scans\1.pdf") == "old_scans__1.pdf"

    assert m._repo_jsonl_path("old_scans.jsonl") == "bench_data/old_scans.jsonl"
    assert (
        m._repo_pdf_path("old_scans/1.pdf")
        == "bench_data/pdfs/old_scans/1.pdf"
    )
    assert (
        m._repo_pdf_path(r"\old_scans\1.pdf")
        == "bench_data/pdfs/old_scans/1.pdf"
    )

    assert (
        m._local_jsonl_path("old_scans.jsonl", work_dir=work_dir)
        == work_dir / "bench_data" / "old_scans.jsonl"
    )
    assert (
        m._local_pdf_path("old_scans/1.pdf", work_dir=work_dir)
        == work_dir / "bench_data" / "pdfs" / "old_scans__1.pdf"
    )


def test_read_jsonl_records_requires_pdf_and_page(tmp_path: Path) -> None:
    path = tmp_path / "old_scans.jsonl"
    write_jsonl(path, [make_text_row()])

    records = m.read_jsonl_records(path)

    assert len(records) == 1
    assert records[0]["pdf"] == "old_scans/1.pdf"
    assert records[0]["page"] == 1

    missing_pdf = tmp_path / "missing_pdf.jsonl"
    write_jsonl(missing_pdf, [{"page": 1, "id": "bad", "type": "present"}])

    with pytest.raises(ValueError, match="pdf"):
        m.read_jsonl_records(missing_pdf)

    missing_page = tmp_path / "missing_page.jsonl"
    write_jsonl(missing_page, [{"pdf": "old_scans/1.pdf", "id": "bad", "type": "present"}])

    with pytest.raises(ValueError, match="page"):
        m.read_jsonl_records(missing_page)


def test_write_jsonl_records_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "subset.jsonl"
    rows = [make_text_row(row_id="a"), make_text_row(row_id="b")]

    m.write_jsonl_records(path, rows)

    loaded = m.read_jsonl_records(path)
    assert [row["id"] for row in loaded] == ["a", "b"]


def test_ordered_unique_pdfs_preserves_jsonl_order() -> None:
    records_by_jsonl = {
        "old_scans.jsonl": [
            make_text_row(pdf="old_scans/1.pdf"),
            make_text_row(pdf="old_scans/2.pdf"),
        ],
        "multi_column.jsonl": [
            make_text_row(pdf="old_scans/1.pdf"),
            make_text_row(pdf="multi_column/a.pdf"),
        ],
    }

    assert m._ordered_unique_pdfs(records_by_jsonl) == [
        "old_scans/1.pdf",
        "old_scans/2.pdf",
        "multi_column/a.pdf",
    ]


def test_sample_prediction_filename_uses_official_flat_alias_pattern(
    tmp_path: Path,
) -> None:
    sample = m.OlmOCRBenchSample(
        sample_id="olmocr_test",
        pdf="old_scans__1.pdf",
        local_pdf_path=tmp_path / "bench_data" / "pdfs" / "old_scans__1.pdf",
        pages=[1],
        source_jsonls=["old_scans.jsonl"],
    )

    assert sample.pdf_stem == "old_scans__1"
    assert sample.prediction_filename == "old_scans__1_pg1_repeat1.md"
    assert sample.prediction_filename_for_page(2) == "old_scans__1_pg2_repeat1.md"
    assert sample.prediction_filename_for_page(2, repeat=3) == (
        "old_scans__1_pg2_repeat3.md"
    )


def test_build_samples_groups_pages_and_source_jsonls(tmp_path: Path) -> None:
    work_dir = tmp_path / "olmocr"
    records_by_jsonl = {
        "old_scans.jsonl": [
            make_text_row(pdf="old_scans/1.pdf", row_id="a"),
            make_text_row(pdf="old_scans/1.pdf", row_id="b"),
        ],
        "multi_column.jsonl": [
            make_text_row(pdf="old_scans/1.pdf", row_id="c"),
            make_text_row(pdf="multi_column/a.pdf", row_id="d"),
        ],
    }

    samples = m._build_samples(
        selected_pdfs=["old_scans/1.pdf", "multi_column/a.pdf"],
        records_by_jsonl=records_by_jsonl,
        work_dir=work_dir,
    )

    assert len(samples) == 2

    first = samples[0]
    assert first.pdf == "old_scans__1.pdf"
    assert first.metadata["original_pdf"] == "old_scans/1.pdf"
    assert first.metadata["local_pdf_alias"] == "old_scans__1.pdf"
    assert first.pages == [1]
    assert first.source_jsonls == ["old_scans.jsonl", "multi_column.jsonl"]
    assert first.local_pdf_path == (
        work_dir / "bench_data" / "pdfs" / "old_scans__1.pdf"
    )
    assert first.metadata["dataset_slug"] == m.DATASET_SLUG


def test_manifest_roundtrip(tmp_path: Path) -> None:
    sample = m.OlmOCRBenchSample(
        sample_id="olmocr_test",
        pdf="old_scans__1.pdf",
        local_pdf_path=tmp_path / "bench_data" / "pdfs" / "old_scans__1.pdf",
        pages=[1],
        source_jsonls=["old_scans.jsonl"],
        metadata={"x": "y"},
    )

    manifest_path = tmp_path / "sample_manifest.jsonl"

    result_path = m.save_manifest([sample], manifest_path)

    assert result_path == manifest_path

    loaded = m.iter_olmocr_samples_from_manifest(manifest_path)

    assert len(loaded) == 1
    assert loaded[0].sample_id == sample.sample_id
    assert loaded[0].pdf == "old_scans__1.pdf"
    assert loaded[0].local_pdf_path == sample.local_pdf_path
    assert loaded[0].pages == [1]
    assert loaded[0].source_jsonls == ["old_scans.jsonl"]
    assert loaded[0].metadata == {"x": "y"}


def test_iter_manifest_respects_limit(tmp_path: Path) -> None:
    samples = [
        m.OlmOCRBenchSample(
            sample_id="a",
            pdf="old_scans__1.pdf",
            local_pdf_path=tmp_path / "a.pdf",
        ),
        m.OlmOCRBenchSample(
            sample_id="b",
            pdf="old_scans__2.pdf",
            local_pdf_path=tmp_path / "b.pdf",
        ),
    ]

    manifest_path = tmp_path / "sample_manifest.jsonl"
    m.save_manifest(samples, manifest_path)

    loaded = m.iter_olmocr_samples_from_manifest(manifest_path, limit=1)

    assert len(loaded) == 1
    assert loaded[0].sample_id == "a"


def test_prepare_olmocr_bench_rewrites_subset_jsonls_and_selects_pdfs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work_dir = tmp_path / "olmocr"

    fake_jsonl_rows = {
        "old_scans.jsonl": [
            make_text_row(pdf="old_scans/1.pdf", row_id="a"),
            make_text_row(pdf="old_scans/2.pdf", row_id="b"),
        ],
        "table_tests.jsonl": [
            make_text_row(pdf="old_scans/1.pdf", row_id="c", test_type="table"),
            make_text_row(pdf="tables/9.pdf", row_id="d", test_type="table"),
        ],
    }

    def fake_jsonl_files_for_track(track: str) -> list[str]:
        assert track == "non_math"
        return ["old_scans.jsonl", "table_tests.jsonl"]

    def fake_download_jsonl(jsonl_name: str, *, work_dir: Path, revision: str) -> Path:
        path = m._local_jsonl_path(jsonl_name, work_dir=work_dir)
        write_jsonl(path, fake_jsonl_rows[jsonl_name])
        return path

    downloaded_pdfs: list[str] = []

    def fake_download_pdf(pdf: str, *, work_dir: Path, revision: str) -> Path:
        downloaded_pdfs.append(pdf)
        path = m._local_pdf_path(pdf, work_dir=work_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-1.4 fake\n")
        return path

    monkeypatch.setattr(m, "jsonl_files_for_track", fake_jsonl_files_for_track)
    monkeypatch.setattr(m, "download_jsonl", fake_download_jsonl)
    monkeypatch.setattr(m, "download_pdf", fake_download_pdf)

    manifest_path = m.prepare_olmocr_bench(
        work_dir=work_dir,
        limit=1,
        track="non_math",
        download_pdfs=True,
    )

    assert manifest_path == work_dir / "sample_manifest.jsonl"

    # Download still uses original official repo path.
    assert downloaded_pdfs == ["old_scans/1.pdf"]

    # Local scoring JSONLs are rewritten to flat aliases for Windows-safe official eval.
    old_scan_records = m.read_jsonl_records(work_dir / "bench_data" / "old_scans.jsonl")
    table_records = m.read_jsonl_records(work_dir / "bench_data" / "table_tests.jsonl")

    assert [row["pdf"] for row in old_scan_records] == ["old_scans__1.pdf"]
    assert [row["pdf"] for row in table_records] == ["old_scans__1.pdf"]

    samples = m.iter_olmocr_samples_from_manifest(manifest_path)
    assert len(samples) == 1
    assert samples[0].pdf == "old_scans__1.pdf"
    assert samples[0].metadata["original_pdf"] == "old_scans/1.pdf"


def test_prepare_olmocr_bench_rejects_invalid_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="limit"):
        m.prepare_olmocr_bench(
            work_dir=tmp_path / "olmocr",
            limit=0,
        )