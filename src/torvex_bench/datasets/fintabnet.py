"""
FinTabNet dataset loader and materializer.

This module prepares the FinTabNet OTSL test split for benchmark runs.
FinTabNet provides table-crop images with table-structure ground truth
(HTML, restored HTML, OTSL, row count, and column count), not full-page PDFs.

Because benchmark adapters consume PDF inputs, each table-crop image is
materialized as a one-page image-only PDF.

Generated local artifacts:
    data/fintabnet/test/images/
    data/fintabnet/test/pdfs/
    data/fintabnet/test/manifest.jsonl

No public manifest is committed to the repo.
Reproducibility is guaranteed by:
    - deterministic sample_id hash (sha1 of content)
    - HF dataset commit hash saved in run summary JSON
    - dataset slug + split locked in code

fintabnet.py is the dataset-side loader/materializer.

It takes official FinTabNet OTSL parquet files and turns them into
benchmark-ready local files: images, PDFs, and runtime manifest.

It normalizes ground-truth fields like HTML, OTSL, rows, cols, span metadata.

It converts FinTabNet table-crop images into one-page PDFs because our
adapters expect PDFs.

Flow (automatic, CLI-controlled):
    prepare_fintabnet(limit=100)
        1. manifest exists with >= 100 rows → return immediately
        2. otherwise → stream from local parquet or HuggingFace → materialize

"""

from __future__ import annotations

import os
import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DATASET_SLUG = "docling-project/FinTabNet_OTSL"
DEFAULT_SPLIT = "test"
EXPECTED_TEST_COUNT = 10397

# env var pattern — same as doclaynet.py
# On Kaggle:  export FINTABNET_RAW_DIR=/kaggle/input/fintabnet-otsl
# On RunPod:  export FINTABNET_RAW_DIR=/workspace/fintabnet_raw
DEFAULT_RAW_DATA_DIR = Path(
    os.getenv("FINTABNET_RAW_DIR", "data/fintabnet_raw")
)
DEFAULT_OUTPUT_DIR = Path(
    os.getenv("FINTABNET_OUTPUT_DIR", "data/fintabnet")
)

# OTSL tokens that indicate merged/spanning table cells.
SPAN_TOKENS = {"lcel", "ucel", "xcel"}


@dataclass(frozen=True)
class FinTabNetSample:
    """
    One benchmark-ready FinTabNet table-crop sample.

    FinTabNet OTSL gives table-crop images, not full document PDFs.
    This object stores:
    - generated one-page PDF path for adapter input
    - ground-truth HTML/OTSL for scoring
    - table metadata for reproducibility
    """

    sample_id: str
    source_index: int
    split: str

    pdf_path: Path
    image_path: Path

    gt_html: str
    gt_html_restored: str
    gt_otsl: str

    rows: int
    cols: int
    has_spans: bool

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
            "image_path": str(self.image_path),
            "gt_html": self.gt_html,
            "gt_html_restored": self.gt_html_restored,
            "gt_otsl": self.gt_otsl,
            "rows": self.rows,
            "cols": self.cols,
            "has_spans": self.has_spans,
            "metadata": self.metadata,
        }


def make_sample_id(
    *,
    split: str,
    source_index: int,
    rows: int,
    cols: int,
    gt_otsl: str,
    gt_html: str,
) -> str:
    """
    Deterministic sample ID — no public manifest needed.

    Two people running prepare_fintabnet() independently on the same
    HuggingFace slug + split will get identical sample IDs.
    """
    digest_source = f"{split}|{source_index}|{rows}|{cols}|{gt_otsl}|{gt_html}"
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:12]

    return f"fintabnet_{split}_{source_index:06d}_{digest}"


def normalize_otsl(value: Any) -> str:
    """
    Normalize the OTSL ground-truth field into a stable string.

    The dataset may return OTSL as a string or as a list of tokens.
    We store it as one clean space-separated string.
    """
    if value is None:
        return ""

    if isinstance(value, list):
        return " ".join(str(item).strip() for item in value if str(item).strip())

    return str(value).strip()


def normalize_html(value: Any) -> str:
    """
    Normalize a FinTabNet HTML ground-truth field into a clean HTML string.

    FinTabNet may store HTML as:
    - a string
    - a list of HTML tokens, e.g. ["<tr>", "<td>", "</td>", "</tr>"]

    We convert token lists into real HTML markup.
    """
    if value is None:
        return ""

    if isinstance(value, list):
        return "".join(str(item) for item in value if str(item))

    return str(value).strip()


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def otsl_tokens(otsl: str) -> list[str]:
    return [
        token.strip()
        for token in str(otsl).replace(",", " ").split()
        if token.strip()
    ]


def detect_spans(otsl: str) -> bool:
    tokens = set(otsl_tokens(otsl))
    return bool(tokens & SPAN_TOKENS)


def coerce_to_pil_image(image_value: Any):
    """
    Convert a raw FinTabNet image value into a PIL image.

    Depending on how the parquet is loaded, the image field may already be
    a PIL image, or it may be a dict containing image bytes.
    """
    import io
    from PIL import Image

    if isinstance(image_value, Image.Image):
        return image_value

    if isinstance(image_value, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_value))

    if isinstance(image_value, dict):
        image_bytes = image_value.get("bytes")
        image_path = image_value.get("path")

        if image_bytes:
            return Image.open(io.BytesIO(image_bytes))

        if image_path:
            return Image.open(image_path)

    raise TypeError(f"Unsupported FinTabNet image value: {type(image_value)!r}")


def save_pil_image(
    *,
    image: Any,
    output_path: str | Path,
) -> None:
    """
    Save a HuggingFace/PIL image as RGB PNG.
    """
    from PIL import Image

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(image)!r}")

    if image.mode in {"RGBA", "LA"}:
        alpha = image.getchannel("A")
        background = Image.new("RGB", image.size, "white")
        background.paste(image.convert("RGB"), mask=alpha)
        image = background
    else:
        image = image.convert("RGB")

    image.save(output_path, format="PNG")


def image_to_pdf(
    *,
    image_path: str | Path,
    pdf_path: str | Path,
) -> None:
    """
    Convert one FinTabNet table-crop PNG into a one-page image-only PDF.

    The extraction adapters consume PDFs, so FinTabNet images must be wrapped
    into PDFs before running the engine.
    """
    import img2pdf

    image_path = Path(image_path)
    pdf_path = Path(pdf_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_bytes = img2pdf.convert(str(image_path))
    pdf_path.write_bytes(pdf_bytes)


def _find_local_parquet_files(
    raw_data_dir: str | Path,
    *,
    split: str = DEFAULT_SPLIT,
) -> list[Path]:
    """
    Find local FinTabNet parquet files using multiple glob patterns.

    Handles HuggingFace snapshot layout and manual download layouts.
    """
    raw_data_dir = Path(raw_data_dir)

    patterns = [
        f"**/{split}-*.parquet",
        f"**/{split}.parquet",
        f"data/{split}-*.parquet",
        f"**/*{split}*.parquet",
    ]

    files: list[Path] = []

    for pattern in patterns:
        files.extend(raw_data_dir.glob(pattern))

    return sorted(set(files))


def load_raw_fintabnet_dataset(
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    *,
    split: str = DEFAULT_SPLIT,
):
    """
    Load FinTabNet dataset for iteration.

    Always uses streaming=True — safe for both limited and full runs.
    Sequential iteration means no benefit to loading into memory.

    Priority:
        local parquet files found → stream from disk
        no local parquet files    → stream directly from HuggingFace
    """
    from datasets import load_dataset

    if split != "test":
        raise ValueError(
            "Only the FinTabNet test split is supported for benchmark runs."
        )

    raw_data_dir = Path(raw_data_dir)
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = _find_local_parquet_files(raw_data_dir, split=split)

    if parquet_files:
        return load_dataset(
            "parquet",
            data_files={split: [str(p) for p in parquet_files]},
            split=split,
            streaming=True,
        )

    # No local files — stream directly from HuggingFace.
    # No snapshot_download needed. Streaming handles it sample by sample.
    return load_dataset(
        DATASET_SLUG,
        split=split,
        cache_dir=str(raw_data_dir),
        streaming=True,
    )


def materialize_fintabnet_sample(
    *,
    raw_sample: dict[str, Any],
    source_index: int,
    split: str = DEFAULT_SPLIT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> FinTabNetSample:
    """
    Convert one raw FinTabNet parquet row into a benchmark-ready sample.

    Creates:
    - PNG image file
    - one-page PDF file
    - FinTabNetSample object with GT fields
    """
    output_dir = Path(output_dir)

    image_dir = output_dir / split / "images"
    pdf_dir = output_dir / split / "pdfs"

    rows = safe_int(raw_sample.get("rows"))
    cols = safe_int(raw_sample.get("cols"))

    gt_otsl = normalize_otsl(raw_sample.get("otsl"))
    gt_html = normalize_html(raw_sample.get("html"))
    gt_html_restored = normalize_html(raw_sample.get("html_restored"))

    if not gt_html and not gt_html_restored:
        raise ValueError(
            f"FinTabNet sample index={source_index} has no HTML ground truth."
        )

    html_for_id = gt_html or gt_html_restored

    sample_id = make_sample_id(
        split=split,
        source_index=source_index,
        rows=rows,
        cols=cols,
        gt_otsl=gt_otsl,
        gt_html=html_for_id,
    )

    image_path = image_dir / f"{sample_id}.png"
    pdf_path = pdf_dir / f"{sample_id}.pdf"

    image_value = raw_sample.get("image")
    if image_value is None:
        raise ValueError(
            f"FinTabNet sample index={source_index} has no image field."
        )

    if not image_path.exists():
        image = coerce_to_pil_image(image_value)
        save_pil_image(image=image, output_path=image_path)

    if not pdf_path.exists():
        image_to_pdf(image_path=image_path, pdf_path=pdf_path)

    sample_has_spans = detect_spans(gt_otsl)

    metadata = {
        "dataset_slug": DATASET_SLUG,
        "split": split,
        "source_index": source_index,
        "rows": rows,
        "cols": cols,
        "has_spans": sample_has_spans,
        "source_format": "table_crop_image",
        "adapter_input": "image_to_pdf",
        "is_full_page_document": False,
    }

    return FinTabNetSample(
        sample_id=sample_id,
        source_index=source_index,
        split=split,
        pdf_path=pdf_path,
        image_path=image_path,
        gt_html=gt_html,
        gt_html_restored=gt_html_restored,
        gt_otsl=gt_otsl,
        rows=rows,
        cols=cols,
        has_spans=sample_has_spans,
        metadata=metadata,
    )


def save_manifest(
    samples: list[FinTabNetSample],
    manifest_path: str | Path,
) -> None:
    """
    Save the local runtime FinTabNet manifest.

    Local artifact only — gitignored.
    Reproducibility anchor is the run summary JSON, not this file.
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


def sample_from_manifest_record(record: dict[str, Any]) -> FinTabNetSample:
    return FinTabNetSample(
        sample_id=str(record["sample_id"]),
        source_index=safe_int(record["source_index"]),
        split=str(record.get("split", DEFAULT_SPLIT)),
        pdf_path=Path(record["pdf_path"]),
        image_path=Path(record["image_path"]),
        gt_html=normalize_html(record.get("gt_html")),
        gt_html_restored=normalize_html(record.get("gt_html_restored")),
        gt_otsl=normalize_otsl(record.get("gt_otsl")),
        rows=safe_int(record.get("rows")),
        cols=safe_int(record.get("cols")),
        has_spans=bool(record.get("has_spans", False)),
        metadata=dict(record.get("metadata", {})),
    )


def iter_fintabnet_samples_from_manifest(
    manifest_path: str | Path,
    *,
    limit: int | None = None,
) -> list[FinTabNetSample]:
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

    limit=None  → need full expected count (10397)
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


def materialize_fintabnet_dataset(
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    *,
    split: str = DEFAULT_SPLIT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
    manifest_path: str | Path | None = None,
) -> list[FinTabNetSample]:
    """
    Materialize FinTabNet samples and save manifest.

    Streams from local parquet or HuggingFace depending on what is available.
    """
    dataset = load_raw_fintabnet_dataset(
        raw_data_dir=raw_data_dir,
        split=split,
    )

    samples: list[FinTabNetSample] = []

    for source_index, raw_sample in enumerate(dataset):
        if limit is not None and len(samples) >= limit:
            break

        sample = materialize_fintabnet_sample(
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


def prepare_fintabnet(
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
        2. Otherwise → materialize via load_raw_fintabnet_dataset()

    load_raw_fintabnet_dataset() handles both cases internally:
        local parquet files found → stream from disk
        no local parquet files    → stream directly from HuggingFace

    Never calls snapshot_download() automatically.
    _download_fintabnet() exists as an explicit opt-in helper only.

    Returns manifest path ready for iter_fintabnet_samples_from_manifest().
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
    # load_raw_fintabnet_dataset() streams from local parquet if present,
    # or streams directly from HuggingFace if not. No explicit download needed.
    materialize_fintabnet_dataset(
        raw_data_dir=raw_data_dir,
        split=split,
        output_dir=output_dir,
        limit=limit,
        manifest_path=manifest_path,
    )

    return manifest_path


def _download_fintabnet(
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
) -> Path:
    """
    Explicit opt-in helper to download the full FinTabNet snapshot locally.

    NOT called automatically by prepare_fintabnet().
    Use this only when you want a full local copy for repeated offline runs.

    On Kaggle/RunPod: set FINTABNET_RAW_DIR and the streaming path handles it.
    """
    from huggingface_hub import snapshot_download

    raw_data_dir = Path(raw_data_dir)
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading FinTabNet OTSL to {raw_data_dir} ...")

    snapshot_download(
        repo_id=DATASET_SLUG,
        repo_type="dataset",
        local_dir=str(raw_data_dir),
    )

    print("Download complete.")

    return raw_data_dir


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