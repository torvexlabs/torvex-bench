"""
runner.py executes extraction jobs.

It receives samples prepared by dataset loaders.

For each sample, it sends the resolved PDF path to the selected adapter.

The adapter runs the engine and returns DocumentResult.

The runner saves:
    - adapter-level DocumentResult JSON
    - normalized scorer-facing JSON
    - per-sample run record
    - append-only run_records.jsonl

The runner does not score, report, convert predictions, or call external
eval harnesses.
"""

from __future__ import annotations

import json
import re
import time
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from torvex_bench.adapters.base import DocumentResult, ExtractionAdapter
from torvex_bench.normalizer import normalize_document


DEFAULT_OUTPUT_DIR = Path("results/raw")


GT_FIELDS_TO_STRIP = {
    # olmOCR
    "tests",
    # OmniDocBench
    "gt_layout_dets",
    "gt_text_blocks",
    "gt_tables",
    "gt_reading_order",
    "gt_bboxes_xyxy",
    "gt_bboxes_raw_poly",
    # FinTabNet / table-style GT variants
    "gt_html",
    "gt_html_restored",
    "gt_otsl",
    # DocLayNet / layout-style GT variants
    "gt_bboxes",
    "gt_bboxes_raw",
    "gt_categories",
    "gt_category_ids",
}


@dataclass(frozen=True, slots=True)
class RunRecord:
    """
    One extraction run record.

    Runner responsibilities:
    - call adapter.extract_document(pdf_path)
    - normalize the DocumentResult
    - save adapter-level + normalized outputs
    - return a small JSON-safe record

    Runner does NOT:
    - score
    - report
    - convert for OmniDocBench
    - call olmOCR/OmniDocBench eval harnesses
    """

    sample_id: str
    dataset: str
    engine: str
    engine_version: str

    pdf_path: str
    status: str
    output_dir: str
    
    input_type: str = "digital"
    raw_output_path: str | None = None
    normalized_output_path: str | None = None
    record_output_path: str | None = None

    page_count: int = 0
    table_count: int = 0
    formula_bbox_count: int = 0

    started_at: str = ""
    finished_at: str = ""
    elapsed_ms: float = 0.0

    error_type: str | None = None
    error_message: str | None = None
    traceback: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "engine": self.engine,
            "engine_version": self.engine_version,
            "pdf_path": self.pdf_path,
            "status": self.status,
            "input_type": self.input_type,
            "output_dir": self.output_dir,
            "raw_output_path": self.raw_output_path,
            "normalized_output_path": self.normalized_output_path,
            "record_output_path": self.record_output_path,
            "page_count": self.page_count,
            "table_count": self.table_count,
            "formula_bbox_count": self.formula_bbox_count,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_ms": self.elapsed_ms,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "traceback": self.traceback,
            "metadata": self.metadata,
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_name(value: str, max_len: int = 96) -> str:
    """
    Make a filesystem-safe ID while keeping it readable.
    """
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")

    if not clean:
        return "sample"

    return clean[:max_len]


def adapter_name(adapter: ExtractionAdapter) -> str:
    return str(getattr(adapter, "name", adapter.__class__.__name__))


def adapter_version(adapter: ExtractionAdapter) -> str:
    return str(getattr(adapter, "version", "unknown"))


def json_safe(value: Any) -> Any:
    """
    Convert common Python objects into JSON-safe values.

    This is intentionally defensive because adapter metadata may contain
    pathlib paths, dataclasses, numpy scalar values, tuples, or other objects.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value):
        return json_safe(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    # numpy arrays, pandas-ish objects, etc.
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass

    # numpy scalar values
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass

    return str(value)


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return path


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(dict(record)), ensure_ascii=False) + "\n")

    return path


def summarize_normalized_document(normalized: dict[str, Any]) -> dict[str, int]:
    pages = normalized.get("pages") or []

    page_count = len(pages)
    table_count = 0
    formula_bbox_count = 0

    for page in pages:
        table_count += len(page.get("tables") or [])
        formula_bbox_count += len(page.get("formula_bboxes") or [])

    return {
        "page_count": page_count,
        "table_count": table_count,
        "formula_bbox_count": formula_bbox_count,
    }


def document_result_to_dict(document: DocumentResult) -> dict[str, Any]:
    """
    Save the adapter-level DocumentResult dataclass as JSON.

    This is not the original raw engine JSON. It is the standardized adapter
    output before normalizer.py converts it into scorer-facing JSON.
    """
    return json_safe(document)


def make_output_paths(
    *,
    output_dir: str | Path,
    dataset: str,
    engine: str,
    sample_id: str,
) -> dict[str, Path]:
    output_dir = Path(output_dir)

    dataset_part = safe_name(dataset)
    engine_part = safe_name(engine)
    sample_part = safe_name(sample_id)

    base_dir = output_dir / dataset_part / engine_part

    return {
        "raw": base_dir / "raw" / f"{sample_part}.json",
        "normalized": base_dir / "normalized" / f"{sample_part}.json",
        "record": base_dir / "records" / f"{sample_part}.json",
        "jsonl": base_dir / "run_records.jsonl",
    }


def run_one_sample(
    *,
    adapter: ExtractionAdapter,
    pdf_path: str | Path,
    sample_id: str,
    dataset: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    input_type: str = "digital",
    sample_metadata: Mapping[str, Any] | None = None,
    run_metadata: Mapping[str, Any] | None = None,
    overwrite: bool = True,
    raise_on_error: bool = False,
) -> RunRecord:
    """
    Run one adapter on one PDF sample.

    This is the core primitive that all dataset-specific runners should call.

    Args:
        adapter:
            ExtractionAdapter implementation.
        pdf_path:
            PDF to extract.
        sample_id:
            Stable sample ID from the dataset loader.
        dataset:
            Dataset name, e.g. "fintabnet", "doclaynet", "olmocr".
        output_dir:
            Root output directory.
        sample_metadata:
            Lightweight dataset-specific metadata copied into the run record.
        run_metadata:
            Run configuration copied into the run record.
        overwrite:
            If False and output already exists, raise FileExistsError.
        raise_on_error:
            If True, re-raise adapter exceptions. Useful in tests/debugging.

    Returns:
        RunRecord
    """
    pdf_path = Path(pdf_path)

    engine = adapter_name(adapter)
    engine_version = adapter_version(adapter)

    paths = make_output_paths(
        output_dir=output_dir,
        dataset=dataset,
        engine=engine,
        sample_id=sample_id,
    )

    if not overwrite:
        existing = [
            path
            for path in (paths["raw"], paths["normalized"], paths["record"])
            if path.exists()
        ]

        if existing:
            raise FileExistsError(
                "Output already exists: "
                + ", ".join(str(path) for path in existing)
            )

    started_at = utc_now_iso()
    t0 = time.perf_counter()

    metadata = {
        "sample": dict(sample_metadata or {}),
        "run": dict(run_metadata or {}),
    }

    try:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        document = adapter.extract_document(str(pdf_path))

        raw_payload = document_result_to_dict(document)
        normalized_payload = normalize_document(document)

        summary = summarize_normalized_document(normalized_payload)

        write_json(paths["raw"], raw_payload)
        write_json(paths["normalized"], normalized_payload)

        finished_at = utc_now_iso()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        record = RunRecord(
            sample_id=str(sample_id),
            dataset=str(dataset),
            engine=engine,
            engine_version=engine_version,
            pdf_path=str(pdf_path),
            status="ok",
            input_type=str(input_type),
            output_dir=str(Path(output_dir)),
            raw_output_path=str(paths["raw"]),
            normalized_output_path=str(paths["normalized"]),
            record_output_path=str(paths["record"]),
            page_count=summary["page_count"],
            table_count=summary["table_count"],
            formula_bbox_count=summary["formula_bbox_count"],
            started_at=started_at,
            finished_at=finished_at,
            elapsed_ms=elapsed_ms,
            metadata=metadata,
        )

        write_json(paths["record"], record.to_manifest_record())
        append_jsonl(paths["jsonl"], record.to_manifest_record())

        return record

    except Exception as exc:
        finished_at = utc_now_iso()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        error_traceback = traceback.format_exc()

        record = RunRecord(
            sample_id=str(sample_id),
            dataset=str(dataset),
            engine=engine,
            engine_version=engine_version,
            pdf_path=str(pdf_path),
            status="error",
            input_type=str(input_type),
            output_dir=str(Path(output_dir)),
            record_output_path=str(paths["record"]),
            started_at=started_at,
            finished_at=finished_at,
            elapsed_ms=elapsed_ms,
            error_type=exc.__class__.__name__,
            error_message=str(exc),
            traceback=error_traceback,
            metadata=metadata,
        )

        write_json(paths["record"], record.to_manifest_record())
        append_jsonl(paths["jsonl"], record.to_manifest_record())

        if raise_on_error:
            raise

        return record


def resolve_sample_pdf_path(
    sample: Any,
    *,
    input_type: str = "digital",
) -> Path:
    """
    Resolve a dataset sample object or manifest record into a PDF path.

    Supports:
    - dataclass/object with .pdf_path field
    - OmniDocBench-style object with .pdf_path(input_type) method
    - dict with "pdf_path"
    - dict with "pdf_path_digital" / "pdf_path_scanned"
    """
    input_type = str(input_type).strip().lower()

    if isinstance(sample, Mapping):
        if input_type == "digital" and "pdf_path_digital" in sample:
            return Path(sample["pdf_path_digital"])

        if input_type == "scanned" and "pdf_path_scanned" in sample:
            return Path(sample["pdf_path_scanned"])

        if "pdf_path" in sample:
            return Path(sample["pdf_path"])

        raise KeyError(
            "Sample dict is missing a PDF path field. Expected one of: "
            "pdf_path, pdf_path_digital, pdf_path_scanned."
        )

    if hasattr(sample, "pdf_path"):
        pdf_path_attr = getattr(sample, "pdf_path")

        if callable(pdf_path_attr):
            return Path(pdf_path_attr(input_type))

        return Path(pdf_path_attr)

    raise TypeError(f"Could not resolve pdf_path from sample type: {type(sample)!r}")


def resolve_sample_id(sample: Any, fallback_index: int) -> str:
    """
    Resolve stable sample ID from a sample object/record.
    """
    if isinstance(sample, Mapping):
        return str(sample.get("sample_id") or f"sample_{fallback_index:06d}")

    if hasattr(sample, "sample_id"):
        return str(getattr(sample, "sample_id"))

    return f"sample_{fallback_index:06d}"


def sample_to_metadata(sample: Any) -> dict[str, Any]:
    """
    Convert a dataset sample object/dict into lightweight metadata.

    Avoid copying huge binary/image fields or ground-truth payloads.
    Dataset manifests remain the source of truth for GT/tests.
    """
    if isinstance(sample, Mapping):
        metadata = dict(sample)
    elif is_dataclass(sample):
        metadata = asdict(sample)
    else:
        metadata = {
            key: getattr(sample, key)
            for key in dir(sample)
            if not key.startswith("_")
            and key not in {"to_manifest_record"}
            and not callable(getattr(sample, key))
        }

    for key in GT_FIELDS_TO_STRIP:
        metadata.pop(key, None)

    return json_safe(metadata)


def run_samples(
    *,
    adapter: ExtractionAdapter,
    samples: Iterable[Any],
    dataset: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    input_type: str = "digital",
    limit: int | None = None,
    run_metadata: Mapping[str, Any] | None = None,
    overwrite: bool = True,
    raise_on_error: bool = False,
) -> list[RunRecord]:
    """
    Run one adapter over many dataset samples.

    Each sample must be either:
    - object/dataclass with .pdf_path field and optional .sample_id
    - OmniDocBench-style object with .pdf_path(input_type) method
    - dict with "pdf_path" and optional "sample_id"
    - dict with "pdf_path_digital" / "pdf_path_scanned"

    Scoring still happens later.
    """
    records: list[RunRecord] = []

    for index, sample in enumerate(samples):
        if limit is not None and len(records) >= limit:
            break

        pdf_path = resolve_sample_pdf_path(sample, input_type=input_type)
        sample_id = resolve_sample_id(sample, fallback_index=index)

        record = run_one_sample(
            adapter=adapter,
            pdf_path=pdf_path,
            sample_id=sample_id,
            dataset=dataset,
            output_dir=output_dir,
            input_type=input_type,
            sample_metadata=sample_to_metadata(sample),
            run_metadata=run_metadata,
            overwrite=overwrite,
            raise_on_error=raise_on_error,
        )

        records.append(record)

    return records


def count_successes(records: Iterable[RunRecord]) -> dict[str, int]:
    """
    Tiny helper for smoke tests / CLI summaries.
    """
    ok = 0
    error = 0

    for record in records:
        if record.status == "ok":
            ok += 1
        else:
            error += 1

    return {
        "ok": ok,
        "error": error,
        "total": ok + error,
    }