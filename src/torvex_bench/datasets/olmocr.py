"""
olmOCR-Bench dataset helper.

Purpose
-------
Prepare the official allenai/olmOCR-bench folder layout for torvex-bench.

Official evaluator expectation
------------------------------
The official evaluator:

    python -m olmocr.bench.benchmark --dir <bench_data> --candidate torvex_extract

expects:

    bench_data/
      *.jsonl
      pdfs/
        <category>/<pdf>.pdf
      torvex_extract/
        <pdf_stem>_pg<page>_repeat1.md

Important
---------
This module does only dataset preparation.

It does NOT:
    - run Torvex Extract
    - normalize output
    - export predictions
    - call the official evaluator
    - compute metrics

Track decision
--------------
Track A is non-math only:

    headers_footers.jsonl
    long_tiny_text.jsonl
    multi_column.jsonl
    old_scans.jsonl
    table_tests.jsonl

Track B / full benchmark can be added later:

    arxiv_math.jsonl
    old_scans_math.jsonl
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


DATASET_SLUG = "allenai/olmOCR-bench"
DATASET_REVISION = "main"

EXPECTED_PDF_COUNT = 1403
EXPECTED_TEST_COUNT = 7019
EXPECTED_NON_MATH_TEST_COUNT = 3634
EXPECTED_MATH_TEST_COUNT = 3385

JSONL_FILES_ALL = [
    "arxiv_math.jsonl",
    "headers_footers.jsonl",
    "long_tiny_text.jsonl",
    "multi_column.jsonl",
    "old_scans.jsonl",
    "old_scans_math.jsonl",
    "table_tests.jsonl",
]

JSONL_FILES_NON_MATH = [
    "headers_footers.jsonl",
    "long_tiny_text.jsonl",
    "multi_column.jsonl",
    "old_scans.jsonl",
    "table_tests.jsonl",
]

JSONL_FILES_MATH = [
    "arxiv_math.jsonl",
    "old_scans_math.jsonl",
]

TRACK_NON_MATH = "non_math"
TRACK_MATH = "math"
TRACK_FULL = "full"

DEFAULT_TRACK = TRACK_NON_MATH

DEFAULT_WORK_DIR = Path(
    os.getenv(
        "OLMOCR_BENCH_WORK_DIR",
        "benchmarks/olmocr/olmOCR_Bench_non_math",
    )
)


@dataclass(slots=True, frozen=True)
class OlmOCRBenchSample:
    """
    One selected olmOCR-Bench PDF sample.

    pdf:
        Official relative PDF path stored in JSONL tests, for example:
            headers_footers/sample_pg1.pdf
            arxiv_math/2502.15977_pg21.pdf

    local_pdf_path:
        Local path under:
            <bench_data>/pdfs/<pdf>

    prediction_filename:
        Official prediction filename expected by benchmark.py:
            <pdf_stem>_pg1_repeat1.md

        Most olmOCR-Bench PDFs are already single-page PDFs.
        The evaluator still uses the JSONL test page number, usually 1.
    """

    sample_id: str
    pdf: str
    local_pdf_path: Path
    pages: list[int] = field(default_factory=list)
    source_jsonls: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pdf_stem(self) -> str:
        return Path(self.pdf).with_suffix("").as_posix()

    def prediction_filename_for_page(self, page: int, repeat: int = 1) -> str:
        return f"{self.pdf_stem}_pg{int(page)}_repeat{int(repeat)}.md"

    @property
    def prediction_filename(self) -> str:
        page = self.pages[0] if self.pages else 1
        return self.prediction_filename_for_page(page)

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "pdf": self.pdf,
            "local_pdf_path": str(self.local_pdf_path),
            "pages": [int(page) for page in self.pages],
            "source_jsonls": list(self.source_jsonls),
            "prediction_filename": self.prediction_filename,
            "metadata": dict(self.metadata),
        }


def jsonl_files_for_track(track: str = DEFAULT_TRACK) -> list[str]:
    if track == TRACK_NON_MATH:
        return list(JSONL_FILES_NON_MATH)

    if track == TRACK_MATH:
        return list(JSONL_FILES_MATH)

    if track == TRACK_FULL:
        return list(JSONL_FILES_ALL)

    raise ValueError("track must be one of: non_math, math, full")


def bench_data_dir(work_dir: str | Path = DEFAULT_WORK_DIR) -> Path:
    return Path(work_dir) / "bench_data"


def default_manifest_path(work_dir: str | Path = DEFAULT_WORK_DIR) -> Path:
    return bench_data_dir(work_dir) / "sample_manifest.jsonl"


def _repo_jsonl_path(jsonl_name: str) -> str:
    return f"bench_data/{jsonl_name}"


def _repo_pdf_path(pdf: str) -> str:
    normalized_pdf = str(pdf).replace("\\", "/").lstrip("/")
    return f"bench_data/pdfs/{normalized_pdf}"


def _local_jsonl_path(jsonl_name: str, *, work_dir: str | Path = DEFAULT_WORK_DIR) -> Path:
    return bench_data_dir(work_dir) / jsonl_name


def _local_pdf_path(pdf: str, *, work_dir: str | Path = DEFAULT_WORK_DIR) -> Path:
    return bench_data_dir(work_dir) / "pdfs" / Path(str(pdf).replace("\\", "/"))


def make_sample_id(pdf: str) -> str:
    digest = hashlib.sha1(pdf.encode("utf-8")).hexdigest()[:12]
    safe_stem = Path(pdf).stem.replace(" ", "_")
    return f"olmocr_{safe_stem}_{digest}"


def _download_file_from_hf(
    repo_path: str,
    local_target: str | Path,
    *,
    revision: str = DATASET_REVISION,
) -> Path:
    """
    Download one file from HuggingFace cache and copy it into bench_data.

    We copy from the HF cache so the generated benchmark folder is simple and
    inspectable.
    """
    from huggingface_hub import hf_hub_download

    local_target = Path(local_target)
    local_target.parent.mkdir(parents=True, exist_ok=True)

    cached_path = hf_hub_download(
        repo_id=DATASET_SLUG,
        filename=repo_path,
        repo_type="dataset",
        revision=revision,
    )

    shutil.copyfile(cached_path, local_target)
    return local_target


def download_jsonl(
    jsonl_name: str,
    *,
    work_dir: str | Path = DEFAULT_WORK_DIR,
    revision: str = DATASET_REVISION,
) -> Path:
    return _download_file_from_hf(
        _repo_jsonl_path(jsonl_name),
        _local_jsonl_path(jsonl_name, work_dir=work_dir),
        revision=revision,
    )


def download_pdf(
    pdf: str,
    *,
    work_dir: str | Path = DEFAULT_WORK_DIR,
    revision: str = DATASET_REVISION,
) -> Path:
    return _download_file_from_hf(
        _repo_pdf_path(pdf),
        _local_pdf_path(pdf, work_dir=work_dir),
        revision=revision,
    )


def read_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            if "pdf" not in record:
                raise ValueError(f"{path}:{line_num} missing required field: pdf")

            if "page" not in record:
                raise ValueError(f"{path}:{line_num} missing required field: page")

            records.append(record)

    return records


def write_jsonl_records(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return path


def _ordered_unique_pdfs(records_by_jsonl: dict[str, list[dict[str, Any]]]) -> list[str]:
    seen: set[str] = set()
    pdfs: list[str] = []

    for records in records_by_jsonl.values():
        for record in records:
            pdf = str(record["pdf"]).replace("\\", "/").lstrip("/")
            if pdf not in seen:
                seen.add(pdf)
                pdfs.append(pdf)

    return pdfs


def _build_samples(
    *,
    selected_pdfs: list[str],
    records_by_jsonl: dict[str, list[dict[str, Any]]],
    work_dir: str | Path = DEFAULT_WORK_DIR,
) -> list[OlmOCRBenchSample]:
    samples: list[OlmOCRBenchSample] = []
    selected_pdf_set = set(selected_pdfs)

    for pdf in selected_pdfs:
        pages: set[int] = set()
        source_jsonls: list[str] = []

        for jsonl_name, records in records_by_jsonl.items():
            matched = False

            for record in records:
                if str(record["pdf"]).replace("\\", "/").lstrip("/") != pdf:
                    continue

                pages.add(int(record["page"]))
                matched = True

            if matched:
                source_jsonls.append(jsonl_name)

        samples.append(
            OlmOCRBenchSample(
                sample_id=make_sample_id(pdf),
                pdf=pdf,
                local_pdf_path=_local_pdf_path(pdf, work_dir=work_dir),
                pages=sorted(pages) or [1],
                source_jsonls=source_jsonls,
                metadata={
                    "dataset_slug": DATASET_SLUG,
                    "dataset_revision": DATASET_REVISION,
                    "selected": pdf in selected_pdf_set,
                },
            )
        )

    return samples


def save_manifest(
    samples: list[OlmOCRBenchSample],
    manifest_path: str | Path,
) -> Path:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as f:
        for rank, sample in enumerate(samples):
            record = sample.to_manifest_record()
            record["rank"] = rank
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return manifest_path


def sample_from_manifest_record(record: dict[str, Any]) -> OlmOCRBenchSample:
    return OlmOCRBenchSample(
        sample_id=str(record["sample_id"]),
        pdf=str(record["pdf"]),
        local_pdf_path=Path(record["local_pdf_path"]),
        pages=[int(page) for page in record.get("pages", [1])],
        source_jsonls=list(record.get("source_jsonls") or []),
        metadata=dict(record.get("metadata") or {}),
    )


def iter_olmocr_samples_from_manifest(
    manifest_path: str | Path,
    *,
    limit: int | None = None,
) -> list[OlmOCRBenchSample]:
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"olmOCR-Bench manifest not found: {manifest_path}")

    samples: list[OlmOCRBenchSample] = []

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(samples) >= limit:
                break

            line = line.strip()
            if not line:
                continue

            samples.append(sample_from_manifest_record(json.loads(line)))

    return samples


def prepare_olmocr_bench(
    *,
    work_dir: str | Path = DEFAULT_WORK_DIR,
    limit: int | None = None,
    track: str = DEFAULT_TRACK,
    download_pdfs: bool = True,
    revision: str = DATASET_REVISION,
    manifest_path: str | Path | None = None,
) -> Path:
    """
    Prepare official olmOCR-Bench data for a smoke/full run.

    Behavior:
        1. Select official JSONL files by track.
        2. Download those JSONLs.
        3. Select first N unique PDFs referenced by those JSONLs.
        4. Rewrite subset JSONLs containing only tests for selected PDFs.
        5. Download selected PDFs.
        6. Write sample_manifest.jsonl.
        7. Return manifest path.

    limit:
        Number of unique PDFs to prepare.
        limit=25 means first 25 unique PDFs from selected JSONLs.

    track:
        non_math:
            excludes arxiv_math.jsonl and old_scans_math.jsonl.

        math:
            only arxiv_math.jsonl and old_scans_math.jsonl.

        full:
            all 7 official JSONL files.
    """
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer or None")

    work_dir = Path(work_dir)
    data_dir = bench_data_dir(work_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    selected_jsonls = jsonl_files_for_track(track)

    records_by_jsonl: dict[str, list[dict[str, Any]]] = {}

    for jsonl_name in selected_jsonls:
        local_path = download_jsonl(
            jsonl_name,
            work_dir=work_dir,
            revision=revision,
        )
        records_by_jsonl[jsonl_name] = read_jsonl_records(local_path)

    all_pdfs = _ordered_unique_pdfs(records_by_jsonl)
    selected_pdfs = all_pdfs[:limit] if limit is not None else all_pdfs
    selected_pdf_set = set(selected_pdfs)

    # Rewrite selected JSONLs in-place so official evaluator scores only the
    # selected PDF subset. This is important for smoke runs.
    for jsonl_name, records in records_by_jsonl.items():
        subset_records = [
            record
            for record in records
            if str(record["pdf"]).replace("\\", "/").lstrip("/") in selected_pdf_set
        ]
        write_jsonl_records(data_dir / jsonl_name, subset_records)

    if download_pdfs:
        for pdf in selected_pdfs:
            local_pdf = _local_pdf_path(pdf, work_dir=work_dir)
            if not local_pdf.exists():
                download_pdf(
                    pdf,
                    work_dir=work_dir,
                    revision=revision,
                )

    samples = _build_samples(
        selected_pdfs=selected_pdfs,
        records_by_jsonl=records_by_jsonl,
        work_dir=work_dir,
    )

    if manifest_path is None:
        manifest_path = default_manifest_path(work_dir)

    save_manifest(samples, manifest_path)
    return Path(manifest_path)