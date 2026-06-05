"""
olmOCR-bench dataset loader and materializer.

olmOCR-bench provides PDF files plus programmatic unit-test assertions.
It is not scored with CER/WER/edit distance. The benchmark scorer should
compute unit-test pass rate only.

This loader prepares benchmark-ready PDF samples and groups all test cases
that belong to the same PDF so runner.py does not extract the same PDF
multiple times.

Important scope decision:
    Torvex v0.1 does not include formula-to-LaTeX extraction.
    Therefore the default subset is "non_math".

Subsets:
    non_math:
        headers_footers.jsonl
        long_tiny_text.jsonl
        multi_column.jsonl
        old_scans.jsonl
        table_tests.jsonl

    math:
        arxiv_math.jsonl
        old_scans_math.jsonl

    all:
        all seven JSONL files

Generated local artifacts:
    data/olmocr_raw/bench_data/*.jsonl
    data/olmocr_raw/bench_data/pdfs/<category>/*.pdf
    data/olmocr/<subset>/manifest.jsonl

Flow:
    prepare_olmocr_bench(subset="non_math", limit=10)
        1. manifest exists and sufficient → return immediately
        2. otherwise → download/read JSONL test files
        3. group test rows by PDF
        4. download only PDFs needed by selected samples
        5. write local runtime manifest

No extraction happens here.
No scoring happens here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DATASET_SLUG = "allenai/olmOCR-bench"

DEFAULT_SUBSET = "non_math"

# Inspected from real JSONL files on 2026-06-05.
# Older project notes said 7,010 tests, but direct JSONL inspection found 7,019.
EXPECTED_TEST_COUNTS = {
    "all": 7019,
    "non_math": 3634,
    "math": 3385,
}

# env var pattern — same as fintabnet.py/doclaynet.py/omnidocbench.py
# On Kaggle: export OLMOCR_RAW_DIR=/kaggle/input/olmocr-bench
# On RunPod: export OLMOCR_RAW_DIR=/workspace/olmocr_raw
DEFAULT_RAW_DATA_DIR = Path(os.getenv("OLMOCR_RAW_DIR", "data/olmocr_raw"))
DEFAULT_OUTPUT_DIR = Path(os.getenv("OLMOCR_OUTPUT_DIR", "data/olmocr"))

TEST_FILES = {
    "arxiv_math": "bench_data/arxiv_math.jsonl",
    "headers_footers": "bench_data/headers_footers.jsonl",
    "long_tiny_text": "bench_data/long_tiny_text.jsonl",
    "multi_column": "bench_data/multi_column.jsonl",
    "old_scans": "bench_data/old_scans.jsonl",
    "old_scans_math": "bench_data/old_scans_math.jsonl",
    "table_tests": "bench_data/table_tests.jsonl",
}

MATH_CATEGORIES = {
    "arxiv_math",
    "old_scans_math",
}

NON_MATH_CATEGORIES = tuple(
    category
    for category in TEST_FILES
    if category not in MATH_CATEGORIES
)

ALL_CATEGORIES = tuple(TEST_FILES.keys())


@dataclass(frozen=True, slots=True)
class OlmOCRTestCase:
    """
    One olmOCR unit-test assertion.

    The schema differs by category:
    - math: math
    - present/absent text: text, case_sensitive, first_n, last_n
    - order: before, after
    - table: cell, up, down, left, right, top_heading, left_heading

    Payload preserves category-specific fields without forcing one fake schema.
    """

    test_id: str
    category: str
    test_type: str
    page: int
    max_diffs: int
    checked: str | None
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "category": self.category,
            "test_type": self.test_type,
            "page": self.page,
            "max_diffs": self.max_diffs,
            "checked": self.checked,
            "payload": self.payload,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class OlmOCRBenchSample:
    """
    One benchmark-ready olmOCR PDF sample.

    A single PDF can have many unit tests.
    Grouping by PDF prevents repeated extraction of the same document.
    """

    sample_id: str
    source_index: int
    subset: str

    pdf_path: Path
    pdf_repo_path: str
    raw_pdf_name: str

    categories: list[str]
    tests: list[OlmOCRTestCase]

    has_math: bool
    test_count: int

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_index": self.source_index,
            "subset": self.subset,
            "pdf_path": str(self.pdf_path),
            "pdf_repo_path": self.pdf_repo_path,
            "raw_pdf_name": self.raw_pdf_name,
            "categories": self.categories,
            "tests": [
                test_case.to_manifest_record()
                for test_case in self.tests
            ],
            "has_math": self.has_math,
            "test_count": self.test_count,
            "metadata": self.metadata,
        }


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def sanitize_id_part(value: str, max_len: int = 48) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")

    if not clean:
        return "unknown"

    return clean[:max_len]


def categories_for_subset(subset: str = DEFAULT_SUBSET) -> tuple[str, ...]:
    """
    Resolve a public subset name into ordered test categories.
    """
    subset = str(subset).strip().lower()

    if subset == "non_math":
        return NON_MATH_CATEGORIES

    if subset == "math":
        return tuple(
            category
            for category in ALL_CATEGORIES
            if category in MATH_CATEGORIES
        )

    if subset == "all":
        return ALL_CATEGORIES

    raise ValueError(
        f"Unknown olmOCR subset={subset!r}. "
        "Expected 'non_math', 'math', or 'all'."
    )


def expected_test_count_for_subset(subset: str = DEFAULT_SUBSET) -> int:
    subset = str(subset).strip().lower()

    if subset not in EXPECTED_TEST_COUNTS:
        raise ValueError(
            f"Unknown olmOCR subset={subset!r}. "
            "Expected 'non_math', 'math', or 'all'."
        )

    return EXPECTED_TEST_COUNTS[subset]


def make_repo_pdf_path(raw_pdf_name: str) -> str:
    """
    Convert JSONL pdf field into HuggingFace repo PDF path.

    Example:
        arxiv_math/2503.04048_pg46.pdf
            → bench_data/pdfs/arxiv_math/2503.04048_pg46.pdf

        tables/b5c5..._pg4.pdf
            → bench_data/pdfs/tables/b5c5..._pg4.pdf
    """
    raw_pdf_name = str(raw_pdf_name).replace("\\", "/").lstrip("/")

    if not raw_pdf_name:
        raise ValueError("olmOCR row has empty pdf field")

    return f"bench_data/pdfs/{raw_pdf_name}"


def make_sample_id(
    *,
    pdf_repo_path: str,
    categories: list[str],
    test_count: int,
) -> str:
    """
    Deterministic sample ID.

    Two people running prepare_olmocr_bench() on the same dataset files
    get the same sample IDs.
    """
    digest_source = f"{pdf_repo_path}|{','.join(categories)}|{test_count}"
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:12]

    stem = sanitize_id_part(Path(pdf_repo_path).stem)

    return f"olmocr_{stem}_{digest}"


def make_test_id(
    *,
    category: str,
    source_index: int,
    row: dict[str, Any],
) -> str:
    raw_id = row.get("id")

    if raw_id:
        return str(raw_id)

    digest = hashlib.sha1(
        json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]

    return f"olmocr_{category}_{source_index:06d}_{digest}"


def download_repo_file(
    *,
    repo_file: str,
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
) -> Path:
    """
    Download one file from the olmOCR HuggingFace dataset if missing.

    Uses local_dir so the folder layout mirrors the HF repo:
        data/olmocr_raw/bench_data/...
    """
    from huggingface_hub import hf_hub_download

    raw_data_dir = Path(raw_data_dir)
    local_path = raw_data_dir / repo_file

    if local_path.exists():
        return local_path

    raw_data_dir.mkdir(parents=True, exist_ok=True)

    downloaded_path = hf_hub_download(
        repo_id=DATASET_SLUG,
        repo_type="dataset",
        filename=repo_file,
        local_dir=str(raw_data_dir),
    )

    return Path(downloaded_path)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL row in {path} at line {line_number}: {exc}"
                ) from exc

    return rows


def normalize_test_case(
    *,
    row: dict[str, Any],
    category: str,
    source_file: str,
    source_index: int,
) -> OlmOCRTestCase:
    """
    Convert one raw JSONL row into an OlmOCRTestCase.

    Category-specific fields are stored in payload.
    """
    if "pdf" not in row:
        raise ValueError(
            f"olmOCR row missing pdf field: category={category}, "
            f"source_index={source_index}"
        )

    if "type" not in row:
        raise ValueError(
            f"olmOCR row missing type field: category={category}, "
            f"source_index={source_index}"
        )

    common_keys = {
        "pdf",
        "page",
        "id",
        "type",
        "max_diffs",
        "checked",
        "url",
    }

    payload = {
        key: value
        for key, value in row.items()
        if key not in common_keys
    }

    metadata = {
        "source_file": source_file,
        "source_index": source_index,
        "url": row.get("url"),
    }

    return OlmOCRTestCase(
        test_id=make_test_id(
            category=category,
            source_index=source_index,
            row=row,
        ),
        category=category,
        test_type=str(row.get("type")),
        page=safe_int(row.get("page"), default=1),
        max_diffs=safe_int(row.get("max_diffs"), default=0),
        checked=(
            None
            if row.get("checked") is None
            else str(row.get("checked"))
        ),
        payload=payload,
        metadata=metadata,
    )


def load_olmocr_test_rows(
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    *,
    subset: str = DEFAULT_SUBSET,
) -> list[tuple[str, dict[str, Any], int]]:
    """
    Download/read JSONL test files for one subset.

    Returns:
        [(category, raw_row, source_index), ...]
    """
    rows: list[tuple[str, dict[str, Any], int]] = []

    for category in categories_for_subset(subset):
        source_file = TEST_FILES[category]

        jsonl_path = download_repo_file(
            repo_file=source_file,
            raw_data_dir=raw_data_dir,
        )

        category_rows = read_jsonl(jsonl_path)

        for source_index, row in enumerate(category_rows):
            rows.append((category, row, source_index))

    return rows


def group_tests_by_pdf(
    raw_rows: list[tuple[str, dict[str, Any], int]],
) -> dict[str, dict[str, Any]]:
    """
    Group unit tests by repo PDF path.

    Output shape:
        {
            "bench_data/pdfs/...pdf": {
                "raw_pdf_name": "...",
                "categories": set(...),
                "tests": [OlmOCRTestCase, ...],
            }
        }
    """
    groups: dict[str, dict[str, Any]] = {}

    for category, row, source_index in raw_rows:
        raw_pdf_name = str(row.get("pdf") or "")
        pdf_repo_path = make_repo_pdf_path(raw_pdf_name)
        source_file = TEST_FILES[category]

        test_case = normalize_test_case(
            row=row,
            category=category,
            source_file=source_file,
            source_index=source_index,
        )

        if pdf_repo_path not in groups:
            groups[pdf_repo_path] = {
                "raw_pdf_name": raw_pdf_name,
                "categories": set(),
                "tests": [],
            }

        groups[pdf_repo_path]["categories"].add(category)
        groups[pdf_repo_path]["tests"].append(test_case)

    return groups


def download_pdf(
    *,
    pdf_repo_path: str,
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
) -> Path:
    """
    Download one benchmark PDF if missing.
    """
    return download_repo_file(
        repo_file=pdf_repo_path,
        raw_data_dir=raw_data_dir,
    )


def materialize_olmocr_sample(
    *,
    pdf_repo_path: str,
    grouped_record: dict[str, Any],
    source_index: int,
    subset: str,
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    download_pdfs: bool = True,
) -> OlmOCRBenchSample:
    """
    Convert one grouped PDF record into an OlmOCRBenchSample.

    If download_pdfs=False, pdf_path is still where the PDF should live,
    but the file is not downloaded. This is useful for unit tests.
    """
    raw_data_dir = Path(raw_data_dir)

    raw_pdf_name = str(grouped_record["raw_pdf_name"])
    categories = sorted(str(c) for c in grouped_record["categories"])
    tests = list(grouped_record["tests"])

    if download_pdfs:
        pdf_path = download_pdf(
            pdf_repo_path=pdf_repo_path,
            raw_data_dir=raw_data_dir,
        )
    else:
        pdf_path = raw_data_dir / pdf_repo_path

    sample_id = make_sample_id(
        pdf_repo_path=pdf_repo_path,
        categories=categories,
        test_count=len(tests),
    )

    has_math = any(category in MATH_CATEGORIES for category in categories)

    metadata = {
        "dataset_slug": DATASET_SLUG,
        "subset": subset,
        "source_index": source_index,
        "source_format": "pdf_plus_unit_tests",
        "adapter_input": "native_pdf",
        "score_metric": "unit_test_pass_rate",
        "is_full_page_document": True,
        "formula_content_scope": (
            "included_in_diagnostic"
            if has_math
            else "not_applicable"
        ),
    }

    return OlmOCRBenchSample(
        sample_id=sample_id,
        source_index=source_index,
        subset=subset,
        pdf_path=pdf_path,
        pdf_repo_path=pdf_repo_path,
        raw_pdf_name=raw_pdf_name,
        categories=categories,
        tests=tests,
        has_math=has_math,
        test_count=len(tests),
        metadata=metadata,
    )


def save_manifest(
    samples: list[OlmOCRBenchSample],
    manifest_path: str | Path,
) -> None:
    """
    Save local runtime manifest.

    Local artifact only — gitignored.
    Reproducibility anchor should be run summary JSON with HF commit hash.
    """
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as f:
        for rank, sample in enumerate(samples):
            record = sample.to_manifest_record()
            record["rank"] = rank
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_manifest(
    manifest_path: str | Path,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    records: list[dict[str, Any]] = []

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(records) >= limit:
                break

            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def test_case_from_manifest_record(record: dict[str, Any]) -> OlmOCRTestCase:
    return OlmOCRTestCase(
        test_id=str(record["test_id"]),
        category=str(record["category"]),
        test_type=str(record["test_type"]),
        page=safe_int(record.get("page"), default=1),
        max_diffs=safe_int(record.get("max_diffs"), default=0),
        checked=(
            None
            if record.get("checked") is None
            else str(record.get("checked"))
        ),
        payload=dict(record.get("payload", {})),
        metadata=dict(record.get("metadata", {})),
    )


def sample_from_manifest_record(record: dict[str, Any]) -> OlmOCRBenchSample:
    tests = [
        test_case_from_manifest_record(test_record)
        for test_record in record.get("tests", [])
    ]

    return OlmOCRBenchSample(
        sample_id=str(record["sample_id"]),
        source_index=safe_int(record.get("source_index"), default=0),
        subset=str(record.get("subset", DEFAULT_SUBSET)),
        pdf_path=Path(record["pdf_path"]),
        pdf_repo_path=str(record["pdf_repo_path"]),
        raw_pdf_name=str(record["raw_pdf_name"]),
        categories=[str(c) for c in record.get("categories", [])],
        tests=tests,
        has_math=bool(record.get("has_math", False)),
        test_count=safe_int(record.get("test_count"), default=len(tests)),
        metadata=dict(record.get("metadata", {})),
    )


def iter_olmocr_samples_from_manifest(
    manifest_path: str | Path,
    *,
    limit: int | None = None,
) -> list[OlmOCRBenchSample]:
    records = load_manifest(
        manifest_path=manifest_path,
        limit=limit,
    )

    return [sample_from_manifest_record(record) for record in records]


def default_manifest_path(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    subset: str = DEFAULT_SUBSET,
) -> Path:
    subset = str(subset).strip().lower()

    return Path(output_dir) / subset / "manifest.jsonl"


def _manifest_counts(manifest_path: str | Path) -> tuple[int, int]:
    """
    Return:
        (sample_count, test_count)
    """
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        return 0, 0

    sample_count = 0
    test_count = 0

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            sample_count += 1

            try:
                record = json.loads(line)
                test_count += safe_int(
                    record.get("test_count"),
                    default=len(record.get("tests", [])),
                )
            except Exception:
                return 0, 0

    return sample_count, test_count


def _manifest_pdfs_exist(manifest_path: str | Path) -> bool:
    """
    Return True only if every PDF referenced by the manifest exists.

    This prevents stale manifests from passing after data/olmocr_raw PDFs
    were manually deleted.
    """
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        return False

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)
                pdf_path = Path(record["pdf_path"])

                if not pdf_path.exists():
                    return False

    except Exception:
        return False

    return True


def _manifest_is_sufficient(
    manifest_path: str | Path,
    *,
    limit: int | None,
    subset: str = DEFAULT_SUBSET,
    require_pdfs: bool = True,
) -> bool:
    """
    Return True if the manifest has enough data for this run.

    limit=None:
        Need expected full test count for subset.
        The manifest has one row per PDF, so we check total test_count,
        not just number of manifest lines.

    limit=N:
        Need at least N PDF samples, not N unit tests.
        One PDF can have multiple tests, so test_count is not predictable
        from limit.
    """
    sample_count, test_count = _manifest_counts(manifest_path)

    if sample_count <= 0:
        return False

    if require_pdfs and not _manifest_pdfs_exist(manifest_path):
        return False

    if limit is not None:
        return sample_count >= limit

    return test_count >= expected_test_count_for_subset(subset)


def materialize_olmocr_bench_dataset(
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    *,
    subset: str = DEFAULT_SUBSET,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
    manifest_path: str | Path | None = None,
    download_pdfs: bool = True,
) -> list[OlmOCRBenchSample]:
    """
    Materialize olmOCR-bench samples and save manifest.

    limit means max PDF samples, not max unit tests.
    This matches runner behavior: one extraction per PDF sample.

    Note:
        The JSONL rows are loaded and grouped before limit is applied.
        This is intentional for simplicity. The full dataset has only 7,019
        test rows, so the memory cost is trivial.
    """
    subset = str(subset).strip().lower()
    categories_for_subset(subset)  # validates early

    raw_rows = load_olmocr_test_rows(
        raw_data_dir=raw_data_dir,
        subset=subset,
    )

    grouped = group_tests_by_pdf(raw_rows)

    samples: list[OlmOCRBenchSample] = []

    for source_index, (pdf_repo_path, grouped_record) in enumerate(grouped.items()):
        if limit is not None and len(samples) >= limit:
            break

        sample = materialize_olmocr_sample(
            pdf_repo_path=pdf_repo_path,
            grouped_record=grouped_record,
            source_index=source_index,
            subset=subset,
            raw_data_dir=raw_data_dir,
            download_pdfs=download_pdfs,
        )

        samples.append(sample)

    if manifest_path is None:
        manifest_path = default_manifest_path(
            output_dir=output_dir,
            subset=subset,
        )

    save_manifest(
        samples=samples,
        manifest_path=manifest_path,
    )

    return samples


def prepare_olmocr_bench(
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    *,
    subset: str = DEFAULT_SUBSET,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
    manifest_path: str | Path | None = None,
    download_pdfs: bool = True,
) -> Path:
    """
    Top-level entry point. This is what runner.py calls.

    Flow:
        1. Manifest exists and is sufficient → return immediately
        2. Otherwise → download/read JSONL test files
        3. Group tests by PDF
        4. Download only referenced PDFs for selected samples
        5. Save manifest

    Default subset is "non_math" because formula-to-LaTeX extraction is
    outside Torvex v0.1 main benchmark scope.

    Use subset="math" or subset="all" for formula/full diagnostics.
    """
    subset = str(subset).strip().lower()
    categories_for_subset(subset)  # validates early

    if manifest_path is None:
        manifest_path = default_manifest_path(
            output_dir=output_dir,
            subset=subset,
        )

    manifest_path = Path(manifest_path)

    if _manifest_is_sufficient(
        manifest_path=manifest_path,
        limit=limit,
        subset=subset,
        require_pdfs=download_pdfs,
    ):
        return manifest_path

    materialize_olmocr_bench_dataset(
        raw_data_dir=raw_data_dir,
        subset=subset,
        output_dir=output_dir,
        limit=limit,
        manifest_path=manifest_path,
        download_pdfs=download_pdfs,
    )

    return manifest_path


def _download_olmocr_bench(
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    *,
    subset: str = DEFAULT_SUBSET,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """
    Explicit opt-in helper to download all JSONL files and PDFs for a subset.

    This is not needed for normal prepare_olmocr_bench() usage because
    prepare_olmocr_bench() already downloads only missing files it needs.

    Included for parity with other dataset loaders.
    """
    prepare_olmocr_bench(
        raw_data_dir=raw_data_dir,
        subset=subset,
        output_dir=output_dir,
        limit=None,
        download_pdfs=True,
    )

    return Path(raw_data_dir)


def get_hf_dataset_commit(slug: str = DATASET_SLUG) -> str:
    """
    Fetch HuggingFace dataset commit hash for run summary reproducibility.

    Never raises — benchmark runs must not fail on a metadata fetch.
    """
    try:
        from huggingface_hub import dataset_info

        info = dataset_info(slug)
        return info.sha or "unknown"
    except Exception as exc:
        print(f"WARNING: Could not fetch HF commit hash for {slug}: {exc}")
        return "unknown"