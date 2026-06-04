"""
DocLayNet dataset loader and materializer.

DocLayNet provides one annotated document page per sample.
Each sample includes:
- raw PDF bytes
- layout bounding boxes
- category IDs
- page metadata

This loader materializes each PDF page to disk and writes a local runtime
manifest for runner.py.

No extraction happens here.
No scoring happens here.
No image-to-PDF conversion is used.

HuggingFace DocLayNet row
        ↓
doclaynet.py reads it
        ↓
writes PDF bytes to data/doclaynet/test/pdfs/
        ↓
normalizes bboxes + category labels
        ↓
writes manifest.jsonl
        ↓
runner.py later reads manifest
        ↓
runner sends pdf_path to Torvex adapter
        ↓
scorer compares Torvex layout zones vs DocLayNet GT

Coordinate space note:
    DocLayNet renders pages onto a 1025×1025 pixel canvas.
    GT bboxes are in this 1025×1025 space.
    Adapter output will be in original PDF point space (e.g. 612×792).
    Coordinate normalization must happen in layout.py before mAP scoring —
    NOT here. This loader preserves GT faithfully.

Negative bbox note:
    Some GT bboxes have coordinates slightly outside the page boundary
    (e.g. y0 = -0.49). This is normal at page edges in DocLayNet.
    Do not clip here. Clip to [0, 0, page_width, page_height] in layout.py.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DATASET_SLUG = "docling-project/DocLayNet-v1.2"
DEFAULT_SPLIT = "test"
EXPECTED_TEST_COUNT = 7613

# FIX: env var pattern — same as fintabnet.py
# On Kaggle: export DOCLAYNET_RAW_DIR=/kaggle/input/doclaynet
# On RunPod: export DOCLAYNET_RAW_DIR=/workspace/doclaynet_raw
DEFAULT_RAW_DATA_DIR = Path(os.getenv("DOCLAYNET_RAW_DIR", "data/doclaynet_raw"))
DEFAULT_OUTPUT_DIR = Path(os.getenv("DOCLAYNET_OUTPUT_DIR", "data/doclaynet"))


# Confirmed correct against real dataset samples.
# Verified from manifest sample run: category_ids [1,7,8,10,5,6,9] all resolve correctly.
# Formula = 3 is correct per DocLayNet paper — low frequency in financial docs is expected.
DOCLAYNET_CATEGORY_MAP = {
    1: "Caption",
    2: "Footnote",
    3: "Formula",
    4: "List-item",
    5: "Page-footer",
    6: "Page-header",
    7: "Picture",
    8: "Section-header",
    9: "Table",
    10: "Text",
    11: "Title",
}


@dataclass(frozen=True, slots=True)
class DocLayNetSample:
    sample_id: str
    source_index: int
    split: str

    pdf_path: Path

    gt_bboxes: list[list[float]]        # xyxy, 1025×1025 space
    gt_bboxes_raw: list[list[float]]    # original coco xywh, preserved for audit
    gt_category_ids: list[int]
    gt_categories: list[str]

    page_width: float                   # always 1025.0 for DocLayNet
    page_height: float                  # always 1025.0 for DocLayNet

    bbox_format: str                    # "xyxy_from_coco_xywh"

    has_table: bool
    has_formula: bool

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_manifest_record(self) -> dict[str, Any]:
        # Explicit field-by-field — no asdict().
        # Path fields serialized as str.
        # Adding a new field here can never create a silent JSON serialization bug.
        return {
            "sample_id": self.sample_id,
            "source_index": self.source_index,
            "split": self.split,
            "pdf_path": str(self.pdf_path),
            "gt_bboxes": self.gt_bboxes,
            "gt_bboxes_raw": self.gt_bboxes_raw,
            "gt_category_ids": self.gt_category_ids,
            "gt_categories": self.gt_categories,
            "page_width": self.page_width,
            "page_height": self.page_height,
            "bbox_format": self.bbox_format,
            "has_table": self.has_table,
            "has_formula": self.has_formula,
            "metadata": self.metadata,
        }


def make_sample_id(
    *,
    split: str,
    source_index: int,
    category_ids: list[int],
    bbox_count: int,
) -> str:
    """
    Deterministic sample ID — no public manifest needed.

    Two people running prepare_doclaynet() independently on the same
    HuggingFace slug + split will get identical sample IDs.
    """
    digest_source = (
        f"{split}|{source_index}|{','.join(map(str, category_ids))}|{bbox_count}"
    )
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:12]

    return f"doclaynet_{split}_{source_index:06d}_{digest}"


def coerce_pdf_bytes(value: Any) -> bytes:
    """
    Convert DocLayNet PDF field into raw bytes.

    HuggingFace may return the pdf column as:
    - bytes          (most common)
    - bytearray
    - {"bytes": ...} (HF image-style dict)
    - base64 string  (rare but possible)
    """
    if isinstance(value, bytes):
        return value

    if isinstance(value, bytearray):
        return bytes(value)

    if isinstance(value, dict):
        raw_bytes = value.get("bytes")

        if isinstance(raw_bytes, bytes):
            return raw_bytes

        if isinstance(raw_bytes, bytearray):
            return bytes(raw_bytes)

        if isinstance(raw_bytes, str):
            return base64.b64decode(raw_bytes)

    if isinstance(value, str):
        return base64.b64decode(value)

    raise TypeError(f"Unsupported DocLayNet pdf field type: {type(value)!r}")


def extract_page_pdf(
    *,
    pdf_bytes: bytes,
    pdf_path: str | Path,
) -> None:
    """
    Write DocLayNet PDF bytes to disk.

    DocLayNet already provides valid PDF bytes per page.
    No img2pdf or PIL conversion needed — just write_bytes.
    """
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(pdf_bytes)


def normalize_doclaynet_bbox(bbox: Any) -> list[float]:
    """
    Convert one DocLayNet COCO-style bbox to xyxy format.

    Input:  [x, y, width, height]   (COCO xywh)
    Output: [x0, y0, x1, y1]        (xyxy)

    Verified correct against real manifest samples:
        raw [210.06, 31.14, 173.98, 39.27]
        out [210.06, 31.14, 384.04, 70.41]
        x1 = 210.06 + 173.98 = 384.04 ✓
        y1 = 31.14  + 39.27  = 70.41  ✓
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"Invalid DocLayNet bbox: {bbox!r}")

    x, y, width, height = [float(v) for v in bbox]

    return [x, y, x + width, y + height]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _page_size_from_metadata(metadata: dict[str, Any]) -> tuple[float, float]:
    """
    Extract page canvas dimensions from DocLayNet metadata.

    Confirmed field names from real samples:
        coco_width  = 1025  (always — DocLayNet renders to 1025×1025)
        coco_height = 1025

    original_width/original_height are the source PDF dimensions (e.g. 612×792).
    Do not use those here — GT bboxes are in coco space, not original PDF space.
    """
    width = (
        metadata.get("coco_width")
        or metadata.get("original_width")
        or 0.0
    )

    height = (
        metadata.get("coco_height")
        or metadata.get("original_height")
        or 0.0
    )

    return safe_float(width), safe_float(height)


def _validate_category_ids(category_ids: list[int]) -> None:
    """
    Raise if any category ID is not in DOCLAYNET_CATEGORY_MAP.

    Silently dropping unknown GT boxes would bias mAP scores and make results
    incomparable to published DocLayNet numbers. If an unknown ID appears it
    means the label map is wrong or the dataset changed — both cases must
    fail loudly so the root cause is fixed, not hidden.
    """
    unknown_ids = sorted(set(category_ids) - set(DOCLAYNET_CATEGORY_MAP))

    if unknown_ids:
        raise ValueError(
            f"Unknown DocLayNet category IDs: {unknown_ids}. "
            "Update DOCLAYNET_CATEGORY_MAP if the dataset label schema changed."
        )


def materialize_doclaynet_sample(
    *,
    raw_sample: dict[str, Any],
    source_index: int,
    split: str = DEFAULT_SPLIT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> DocLayNetSample:
    """
    Convert one raw DocLayNet dataset row into a benchmark-ready sample.

    Steps:
    - read bboxes + category_ids
    - filter unknown category IDs (warn, never crash)
    - validate category IDs and fail loudly if unknown
    - make deterministic sample_id
    - write PDF bytes to disk (skip if already exists)
    - convert bboxes coco xywh → xyxy
    - read page dimensions from metadata
    - warn if page dimensions are zero
    - return DocLayNetSample
    """
    output_dir = Path(output_dir)

    raw_bboxes = raw_sample.get("bboxes") or []
    raw_category_ids = [int(v) for v in raw_sample.get("category_id") or []]

    category_ids = raw_category_ids
    _validate_category_ids(category_ids)

    if len(raw_bboxes) != len(category_ids):
        raise ValueError(
            "DocLayNet bbox/category length mismatch: "
            f"bboxes={len(raw_bboxes)}, category_ids={len(category_ids)}, "
            f"source_index={source_index}"
        )

    sample_id = make_sample_id(
        split=split,
        source_index=source_index,
        category_ids=category_ids,
        bbox_count=len(raw_bboxes),
    )

    pdf_path = output_dir / split / "pdfs" / f"{sample_id}.pdf"

    pdf_value = raw_sample.get("pdf")
    if pdf_value is None:
        raise ValueError(
            f"DocLayNet sample index={source_index} has no pdf field."
        )

    if not pdf_path.exists():
        pdf_bytes = coerce_pdf_bytes(pdf_value)
        extract_page_pdf(pdf_bytes=pdf_bytes, pdf_path=pdf_path)

    gt_bboxes_raw = [
        [float(v) for v in bbox]
        for bbox in raw_bboxes
    ]

    gt_bboxes = [
        normalize_doclaynet_bbox(bbox)
        for bbox in raw_bboxes
    ]

    gt_categories = [
        DOCLAYNET_CATEGORY_MAP[cid]
        for cid in category_ids
    ]

    metadata = dict(raw_sample.get("metadata") or {})
    page_width, page_height = _page_size_from_metadata(metadata)

    # FIX: warn when page dimensions are zero instead of silently using 0.0
    if page_width == 0.0 and page_height == 0.0:
        print(
            f"WARNING: page dimensions are 0.0 for source_index={source_index}. "
            f"Metadata keys seen: {sorted(metadata.keys())}"
        )

    has_table = 9 in category_ids
    has_formula = 3 in category_ids

    metadata.update(
        {
            "dataset_slug": DATASET_SLUG,
            "split": split,
            "source_index": source_index,
            "source_format": "pdf_page",
            "adapter_input": "pdf_bytes",
            "bbox_input_format": "coco_xywh",
            "bbox_output_format": "xyxy",
            "field_names_seen": sorted(raw_sample.keys()),
        }
    )

    return DocLayNetSample(
        sample_id=sample_id,
        source_index=source_index,
        split=split,
        pdf_path=pdf_path,
        gt_bboxes=gt_bboxes,
        gt_bboxes_raw=gt_bboxes_raw,
        gt_category_ids=category_ids,
        gt_categories=gt_categories,
        page_width=page_width,
        page_height=page_height,
        bbox_format="xyxy_from_coco_xywh",
        has_table=has_table,
        has_formula=has_formula,
        metadata=metadata,
    )


def save_manifest(
    samples: list[DocLayNetSample],
    manifest_path: str | Path,
) -> None:
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


def sample_from_manifest_record(record: dict[str, Any]) -> DocLayNetSample:
    return DocLayNetSample(
        sample_id=str(record["sample_id"]),
        source_index=int(record["source_index"]),
        split=str(record.get("split", DEFAULT_SPLIT)),
        pdf_path=Path(record["pdf_path"]),
        gt_bboxes=[
            [float(v) for v in bbox]
            for bbox in record.get("gt_bboxes", [])
        ],
        gt_bboxes_raw=[
            [float(v) for v in bbox]
            for bbox in record.get("gt_bboxes_raw", [])
        ],
        gt_category_ids=[
            int(v) for v in record.get("gt_category_ids", [])
        ],
        gt_categories=[
            str(v) for v in record.get("gt_categories", [])
        ],
        page_width=safe_float(record.get("page_width")),
        page_height=safe_float(record.get("page_height")),
        bbox_format=str(record.get("bbox_format", "xyxy_from_coco_xywh")),
        has_table=bool(record.get("has_table", False)),
        has_formula=bool(record.get("has_formula", False)),
        metadata=dict(record.get("metadata", {})),
    )


def iter_doclaynet_samples_from_manifest(
    manifest_path: str | Path,
    *,
    limit: int | None = None,
) -> list[DocLayNetSample]:
    records = load_manifest(
        manifest_path=manifest_path,
        limit=limit,
    )

    return [sample_from_manifest_record(record) for record in records]


def default_manifest_path(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    split: str = DEFAULT_SPLIT,
) -> Path:
    return Path(output_dir) / split / "manifest.jsonl"


def _manifest_is_sufficient(
    manifest_path: str | Path,
    limit: int | None,
    expected_full_count: int = EXPECTED_TEST_COUNT,
) -> bool:
    """
    Return True if the manifest has enough rows for this run.

    limit=None  → need full expected count (7613)
    limit=N     → need at least N rows
    """
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        return False

    with manifest_path.open("r", encoding="utf-8") as f:
        count = sum(1 for line in f if line.strip())

    if limit is None:
        return count >= expected_full_count

    return count >= limit


def _download_doclaynet(
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
) -> Path:
    """
    Download DocLayNet from HuggingFace into a local folder.

    Explicit manual helper only. Not called by prepare_doclaynet().
    """
    from huggingface_hub import snapshot_download

    raw_data_dir = Path(raw_data_dir)
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading DocLayNet to {raw_data_dir} ...")

    snapshot_download(
        repo_id=DATASET_SLUG,
        repo_type="dataset",
        local_dir=str(raw_data_dir),
        # FIX: removed local_dir_use_symlinks=False — deprecated in
        # huggingface_hub >= 0.23, raises TypeError on recent versions.
        # Default behaviour with local_dir set is already no symlinks.
    )

    print("Download complete.")

    return raw_data_dir


def _find_local_parquet_files(
    raw_data_dir: str | Path,
    *,
    split: str = DEFAULT_SPLIT,
) -> list[Path]:
    raw_data_dir = Path(raw_data_dir)

    patterns = [
        f"**/{split}-*.parquet",
        f"**/{split}.parquet",
        f"**/*{split}*.parquet",
    ]

    files: list[Path] = []

    for pattern in patterns:
        files.extend(raw_data_dir.glob(pattern))

    return sorted(set(files))


def load_raw_doclaynet_dataset(
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    *,
    split: str = DEFAULT_SPLIT,
):
    """
    Load DocLayNet dataset for iteration.

    Always uses streaming=True.

    FIX: was streaming=limit is not None which loaded the entire dataset
    into memory for full runs (7613 pages of embedded PDFs). streaming=True
    is safe for both limited and full runs — iteration is always sequential.
    """
    from datasets import load_dataset

    raw_data_dir = Path(raw_data_dir)
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = _find_local_parquet_files(raw_data_dir, split=split)

    if parquet_files:
        return load_dataset(
            "parquet",
            data_files={split: [str(p) for p in parquet_files]},
            split=split,
            streaming=True,     # FIX: always stream
        )

    return load_dataset(
        DATASET_SLUG,
        split=split,
        cache_dir=str(raw_data_dir),
        streaming=True,         # FIX: always stream
    )


def materialize_doclaynet_dataset(
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    *,
    split: str = DEFAULT_SPLIT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
    manifest_path: str | Path | None = None,
) -> list[DocLayNetSample]:
    # FIX: was misindented — the load call closing paren was at wrong indent level
    dataset = load_raw_doclaynet_dataset(
        raw_data_dir=raw_data_dir,
        split=split,
    )

    samples: list[DocLayNetSample] = []

    for source_index, raw_sample in enumerate(dataset):
        if limit is not None and len(samples) >= limit:
            break

        sample = materialize_doclaynet_sample(
            raw_sample=raw_sample,
            source_index=source_index,
            split=split,
            output_dir=output_dir,
        )

        samples.append(sample)

    if manifest_path is None:
        manifest_path = default_manifest_path(
            output_dir=output_dir,
            split=split,
        )

    save_manifest(
        samples=samples,
        manifest_path=manifest_path,
    )

    return samples


def prepare_doclaynet(
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    *,
    split: str = DEFAULT_SPLIT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
    manifest_path: str | Path | None = None,
) -> Path:
    """
    Top-level entry point. This is what runner.py calls.

    Flow:
        1. Manifest exists and has enough rows → return immediately
        2. Otherwise → materialize via load_raw_doclaynet_dataset()

    load_raw_doclaynet_dataset() handles both cases internally:
        local parquet files found → stream from disk
        no local parquet files    → stream directly from HuggingFace

    Never calls snapshot_download() automatically.
    _download_doclaynet() exists as an explicit opt-in helper only.
    Calling snapshot_download() here would download the full repo
    before a single sample is processed, defeating streaming entirely.

    Returns manifest path ready for iter_doclaynet_samples_from_manifest().
    """
    if manifest_path is None:
        manifest_path = default_manifest_path(
            output_dir=output_dir,
            split=split,
        )

    manifest_path = Path(manifest_path)

    # Step 1 — manifest already sufficient
    if _manifest_is_sufficient(manifest_path=manifest_path, limit=limit):
        return manifest_path

    # Step 2 — materialize
    # load_raw_doclaynet_dataset() streams from local parquet if present,
    # or streams directly from HuggingFace if not. No explicit download needed.
    materialize_doclaynet_dataset(
        raw_data_dir=raw_data_dir,
        split=split,
        output_dir=output_dir,
        limit=limit,
        manifest_path=manifest_path,
    )

    return manifest_path


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
        # FIX: was silent — now prints so you know when it fails
        print(f"WARNING: Could not fetch HF commit hash for {slug}: {exc}")
        return "unknown"