"""
OmniDocBench dataset helper.

Locked benchmark decision
-------------------------
For torvex-bench, OmniDocBench is treated as the official scanned/image-page
benchmark only.

Current official HuggingFace main branch provides:

    OmniDocBench.json
    images/<page-image>

It does not expose pdfs/ or ori_pdfs/ on main. We therefore do not use
OmniDocBench as the official digital-PDF benchmark.

This module does only dataset preparation:

    1. Load OmniDocBench.json.
    2. Download/materialize needed page images.
    3. Build a small JSONL manifest.
    4. Return image samples to the prediction harness.

No extraction happens here.
No normalization happens here.
No scoring happens here.
No official evaluator is called here.

The later harness does this:

    image page
        -> temporary single-page scanned PDF
        -> TorvexExtractAdapter
        -> normalize_document()
        -> exporters/omnidocbench_markdown.py writes <image_stem>.md
        -> official_omnidocbench.py calls omnidocbench-eval

Important filename rule
-----------------------
OmniDocBench GT stores only the image filename in page_info.image_path:

    page-d1561665-5359-42fe-920c-d6e3bff81953.png

The actual HuggingFace repo path is:

    images/page-d1561665-5359-42fe-920c-d6e3bff81953.png

The official evaluator expects prediction markdown named by the same stem:

    page-d1561665-5359-42fe-920c-d6e3bff81953.md
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DATASET_SLUG = "opendatalab/OmniDocBench"
DATASET_REVISION = "main"
EXPECTED_COUNT = 1651

GT_JSON_FILENAME = "OmniDocBench.json"
IMAGE_SUBDIR = "images"

# env var pattern — same style as the other dataset helpers.
# On Kaggle:
#   export OMNIDOCBENCH_RAW_DIR=/kaggle/input/omnidocbench
# On RunPod:
#   export OMNIDOCBENCH_RAW_DIR=/workspace/omnidocbench_raw
DEFAULT_RAW_DATA_DIR = Path(os.getenv(
    "OMNIDOCBENCH_RAW_DIR",
    "benchmarks/omnidocbench/OmniDocBench_scanned/gt_dataset",
))

DEFAULT_OUTPUT_DIR = Path(os.getenv(
    "OMNIDOCBENCH_OUTPUT_DIR",
    "benchmarks/omnidocbench/OmniDocBench_scanned/gt_dataset",
))

# Keep a single explicit mode name for CLI/report clarity.
# This is NOT a digital-vs-scanned switch.
INPUT_TYPE_SCANNED = "scanned"
DEFAULT_INPUT_TYPE = INPUT_TYPE_SCANNED


@dataclass(slots=True)
class OmniDocBenchSample:
    """
    One OmniDocBench page sample.

    This sample is image-based because current official OmniDocBench main
    stores page images, not PDF folders.

    Fields:
        source_index:
            Index of the page record inside OmniDocBench.json.

        sample_id:
            Stable torvex-bench id. Deterministic from source_index and
            image filename.

        image_filename:
            The filename stored in GT page_info.image_path, for example:
            page-d1561665-5359-42fe-920c-d6e3bff81953.png

        image_repo_path:
            Path inside the HuggingFace dataset repo:
            images/<image_filename>

        image_path:
            Local downloaded image path.

        page_info, layout_dets, extra:
            Raw GT fields preserved for audit and for writing subset GT JSON.
            The official OmniDocBench evaluator reads these fields directly.
            torvex-bench does not implement the metric logic.
    """

    source_index: int
    sample_id: str
    image_filename: str
    image_repo_path: str
    image_path: Path

    page_info: dict[str, Any]
    layout_dets: list[dict[str, Any]]
    extra: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def image_stem(self) -> str:
        """Return filename stem used by the official prediction .md file."""
        return Path(self.image_filename).stem

    @property
    def prediction_filename(self) -> str:
        """
        Return official end-to-end markdown filename.

        Example:
            page-d156...png -> page-d156...md
        """
        return f"{self.image_stem}.md"

    def to_manifest_record(self) -> dict[str, Any]:
        """Convert this sample into one JSONL-safe manifest record."""
        return {
            "source_index": int(self.source_index),
            "sample_id": self.sample_id,
            "image_filename": self.image_filename,
            "image_repo_path": self.image_repo_path,
            "image_path": str(self.image_path),
            "page_info": self.page_info,
            "layout_dets": self.layout_dets,
            "extra": self.extra,
            "metadata": self.metadata,
        }


def make_sample_id(*, source_index: int, image_filename: str) -> str:
    """
    Build a stable sample id.

    Two machines preparing the same OmniDocBench page should produce the same
    id, independent of local cache paths.
    """
    digest_source = f"{source_index}|{image_filename}"
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:12]
    return f"omnidocbench_{source_index:06d}_{digest}"


def image_repo_path(image_filename: str) -> str:
    """
    Return the HuggingFace repo path for an OmniDocBench image.

    GT usually stores only the filename. If a future JSON already includes
    images/<name>, keep it as-is.
    """
    normalized = str(image_filename).replace("\\", "/").lstrip("/")
    if normalized.startswith(f"{IMAGE_SUBDIR}/"):
        return normalized
    return f"{IMAGE_SUBDIR}/{normalized}"


def image_filename_from_page_info(page_info: dict[str, Any]) -> str:
    """Extract page_info.image_path and normalize it to a filename."""
    raw_image_path = str(page_info.get("image_path") or "")
    if not raw_image_path:
        raise ValueError("OmniDocBench record has no page_info.image_path.")

    return Path(raw_image_path.replace("\\", "/")).name


def local_image_path(
    image_filename: str,
    *,
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
) -> Path:
    """Return local path for one downloaded OmniDocBench image."""
    return Path(raw_data_dir) / IMAGE_SUBDIR / image_filename


def default_manifest_path(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Return default local manifest path for prepared OmniDocBench samples."""
    return Path(output_dir) / "sample_manifest.jsonl"


def _gt_json_path(raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR) -> Path:
    """Return expected local path for OmniDocBench.json."""
    return Path(raw_data_dir) / GT_JSON_FILENAME


def load_gt_json(raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR) -> list[dict[str, Any]]:
    """
    Load OmniDocBench.json from local raw_data_dir.

    Raises a clear error if the file is missing.
    """
    gt_path = _gt_json_path(raw_data_dir)

    if not gt_path.exists():
        raise FileNotFoundError(
            f"OmniDocBench.json not found at {gt_path}. "
            "Run prepare_omnidocbench() to download metadata, or set "
            "OMNIDOCBENCH_RAW_DIR to a folder containing OmniDocBench.json."
        )

    data = json.loads(gt_path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise TypeError(
            f"OmniDocBench.json must be a list of page records. "
            f"Got {type(data).__name__}."
        )

    return data


def _download_file_from_hf(
    repo_path: str,
    *,
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    revision: str = DATASET_REVISION,
) -> Path:
    """
    Download one file from HuggingFace and copy it into raw_data_dir.

    hf_hub_download stores files in the HF cache. We copy into data/ so the
    benchmark folder has a simple, inspectable layout:

        data/omnidocbench_raw/OmniDocBench.json
        data/omnidocbench_raw/images/<image>.png
    """
    from huggingface_hub import hf_hub_download

    raw_data_dir = Path(raw_data_dir)
    local_target = raw_data_dir / repo_path
    local_target.parent.mkdir(parents=True, exist_ok=True)

    cached_path = hf_hub_download(
        repo_id=DATASET_SLUG,
        filename=repo_path,
        repo_type="dataset",
        revision=revision,
    )

    shutil.copyfile(cached_path, local_target)
    return local_target


def download_omnidocbench_metadata(
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    *,
    revision: str = DATASET_REVISION,
) -> Path:
    """
    Download OmniDocBench.json only.

    This intentionally does not download all images. For smoke runs we download
    only the images needed by the selected limit.
    """
    return _download_file_from_hf(
        GT_JSON_FILENAME,
        raw_data_dir=raw_data_dir,
        revision=revision,
    )


def download_omnidocbench_image(
    image_filename: str,
    *,
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    revision: str = DATASET_REVISION,
) -> Path:
    """
    Download one OmniDocBench page image.

    Input:
        page-d1561665-5359-42fe-920c-d6e3bff81953.png

    Downloads:
        images/page-d1561665-5359-42fe-920c-d6e3bff81953.png
    """
    return _download_file_from_hf(
        image_repo_path(image_filename),
        raw_data_dir=raw_data_dir,
        revision=revision,
    )


def poly_to_xyxy(poly: list[float]) -> list[float]:
    """
    Convert OmniDocBench quadrilateral poly to axis-aligned xyxy bbox.

    This is kept as a small audit/helper utility. The official end-to-end
    markdown path does not use bbox metrics.
    """
    if len(poly) != 8:
        raise ValueError(f"OmniDocBench poly must have 8 floats, got {len(poly)}: {poly!r}")

    xs = [float(poly[i]) for i in range(0, 8, 2)]
    ys = [float(poly[i]) for i in range(1, 8, 2)]

    return [min(xs), min(ys), max(xs), max(ys)]


def materialize_omnidocbench_sample(
    *,
    raw_record: dict[str, Any],
    source_index: int,
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    require_image_exists: bool = True,
) -> OmniDocBenchSample:
    """
    Convert one raw OmniDocBench JSON record into a benchmark sample.

    This does not download. prepare_omnidocbench() handles download first.

    What this proves when it succeeds:
        - page_info.image_path exists.
        - local images/<filename> exists when require_image_exists=True.
        - sample id and prediction filename are deterministic.
    """
    raw_data_dir = Path(raw_data_dir)

    page_info = dict(raw_record.get("page_info") or {})
    layout_dets = list(raw_record.get("layout_dets") or [])
    extra = dict(raw_record.get("extra") or {})

    image_filename = image_filename_from_page_info(page_info)
    local_path = local_image_path(image_filename, raw_data_dir=raw_data_dir)

    if require_image_exists and not local_path.exists():
        raise FileNotFoundError(
            f"Missing OmniDocBench image for source_index={source_index}: {local_path}. "
            "Run prepare_omnidocbench() with download_images=True."
        )

    sample_id = make_sample_id(
        source_index=source_index,
        image_filename=image_filename,
    )

    metadata = {
        "dataset_slug": DATASET_SLUG,
        "dataset_revision": DATASET_REVISION,
        "source_index": int(source_index),
        "image_filename": image_filename,
        "image_repo_path": image_repo_path(image_filename),
        "prediction_filename": f"{Path(image_filename).stem}.md",
        "page_width": page_info.get("width"),
        "page_height": page_info.get("height"),
        "field_names_seen": sorted(raw_record.keys()),
    }

    return OmniDocBenchSample(
        source_index=source_index,
        sample_id=sample_id,
        image_filename=image_filename,
        image_repo_path=image_repo_path(image_filename),
        image_path=local_path,
        page_info=page_info,
        layout_dets=layout_dets,
        extra=extra,
        metadata=metadata,
    )


def save_manifest(
    samples: list[OmniDocBenchSample],
    manifest_path: str | Path,
) -> Path:
    """Write one JSONL manifest record per prepared sample."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as f:
        for rank, sample in enumerate(samples):
            record = sample.to_manifest_record()
            record["rank"] = rank
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return manifest_path


def sample_from_manifest_record(record: dict[str, Any]) -> OmniDocBenchSample:
    """Restore one OmniDocBenchSample from a manifest JSONL record."""
    image_filename = str(record["image_filename"])

    return OmniDocBenchSample(
        source_index=int(record["source_index"]),
        sample_id=str(record["sample_id"]),
        image_filename=image_filename,
        image_repo_path=str(record.get("image_repo_path") or image_repo_path(image_filename)),
        image_path=Path(record["image_path"]),
        page_info=dict(record.get("page_info") or {}),
        layout_dets=list(record.get("layout_dets") or []),
        extra=dict(record.get("extra") or {}),
        metadata=dict(record.get("metadata") or {}),
    )


def iter_omnidocbench_samples_from_manifest(
    manifest_path: str | Path,
    *,
    limit: int | None = None,
) -> list[OmniDocBenchSample]:
    """
    Load prepared samples from a manifest.

    This is the normal runtime path after prepare_omnidocbench().
    """
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"OmniDocBench manifest not found: {manifest_path}")

    samples: list[OmniDocBenchSample] = []

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(samples) >= limit:
                break

            line = line.strip()
            if not line:
                continue

            samples.append(sample_from_manifest_record(json.loads(line)))

    return samples


def materialize_omnidocbench_dataset(
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
    manifest_path: str | Path | None = None,
    download_images: bool = True,
    revision: str = DATASET_REVISION,
) -> list[OmniDocBenchSample]:
    """
    Materialize OmniDocBench samples from local GT JSON and page images.

    This is image-based. It does not expect pdfs/ or ori_pdfs/.

    limit:
        Small smoke runs should pass limit=1, 3, 25, etc.
        Full official scanned/image benchmark can pass limit=None.

    download_images:
        True downloads missing selected images.
        False requires images to already exist locally.
    """
    raw_data_dir = Path(raw_data_dir)

    gt_records = load_gt_json(raw_data_dir)
    selected_records = gt_records[:limit] if limit is not None else gt_records

    samples: list[OmniDocBenchSample] = []

    for source_index, raw_record in enumerate(selected_records):
        page_info = dict(raw_record.get("page_info") or {})
        image_filename = image_filename_from_page_info(page_info)
        local_path = local_image_path(image_filename, raw_data_dir=raw_data_dir)

        if download_images and not local_path.exists():
            download_omnidocbench_image(
                image_filename,
                raw_data_dir=raw_data_dir,
                revision=revision,
            )

        sample = materialize_omnidocbench_sample(
            raw_record=raw_record,
            source_index=source_index,
            raw_data_dir=raw_data_dir,
            require_image_exists=True,
        )
        samples.append(sample)

    if manifest_path is None:
        manifest_path = default_manifest_path(output_dir=output_dir)

    save_manifest(samples, manifest_path)
    return samples


def prepare_omnidocbench(
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
    manifest_path: str | Path | None = None,
    download_images: bool = True,
    revision: str = DATASET_REVISION,
) -> Path:
    """
    Prepare OmniDocBench current-main image samples.

    Command-style behavior:
        1. If OmniDocBench.json is missing, download it.
        2. Select first limit records, or all records when limit=None.
        3. Download selected images when download_images=True.
        4. Write manifest.jsonl.
        5. Return manifest path.

    This function is intentionally scanned/image-only.
    """
    raw_data_dir = Path(raw_data_dir)

    if manifest_path is None:
        manifest_path = default_manifest_path(output_dir=output_dir)

    if not _gt_json_path(raw_data_dir).exists():
        download_omnidocbench_metadata(
            raw_data_dir=raw_data_dir,
            revision=revision,
        )

    materialize_omnidocbench_dataset(
        raw_data_dir=raw_data_dir,
        output_dir=output_dir,
        limit=limit,
        manifest_path=manifest_path,
        download_images=download_images,
        revision=revision,
    )

    return Path(manifest_path)