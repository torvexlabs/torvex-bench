from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DATASET_SLUG = "docling-project/FinTabNet_OTSL"
DEFAULT_SPLIT = "test"

# Where you downloaded the raw HuggingFace test parquet files.
DEFAULT_RAW_DATA_DIR = Path("C:/datasets/fintabnet_otsl_raw")

# Where our benchmark will create clean local images, PDFs, and manifest.
DEFAULT_OUTPUT_DIR = Path("data/fintabnet")

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
        record = asdict(self)
        record["pdf_path"] = str(self.pdf_path)
        record["image_path"] = str(self.image_path)
        return record
    

def find_test_parquet_files(
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
) -> list[Path]:
    """
    Find the downloaded FinTabNet OTSL test parquet files.

    Expected raw folder:
        C:/datasets/fintabnet_otsl_raw/data/test-*.parquet
    """
    raw_data_dir = Path(raw_data_dir)
    data_dir = raw_data_dir / "data"

    if not data_dir.exists():
        raise FileNotFoundError(
            f"FinTabNet raw data folder not found: {data_dir}. "
            "Expected downloaded parquet files under raw_data_dir/data/"
        )

    parquet_files = sorted(data_dir.glob("test-*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No FinTabNet test parquet files found in {data_dir}. "
            "Expected files like test-00000-of-00002-*.parquet"
        )

    return parquet_files


def load_raw_fintabnet_dataset(
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    *,
    split: str = DEFAULT_SPLIT,
):
    """
    Load local FinTabNet OTSL parquet files as a HuggingFace Dataset.

    This reads from the raw parquet files downloaded outside the repo, for example:
        C:/datasets/fintabnet_otsl_raw/data/test-*.parquet

    It does not download anything from HuggingFace.
    """
    if split != "test":
        raise ValueError(
            "Only the FinTabNet test split is supported for benchmark runs."
        )

    parquet_files = find_test_parquet_files(raw_data_dir)

    from datasets import load_dataset

    data_files = {
        split: [str(path) for path in parquet_files],
    }

    return load_dataset(
        "parquet",
        data_files=data_files,
        split=split,
    )


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
        return "".join(str(item).strip() for item in value if str(item).strip())

    return str(value).strip()


def safe_int(value: Any) -> int:
    """
    Convert a dataset value into int safely.

    FinTabNet rows/cols should be numbers, but this keeps the loader
    stable if parquet returns strings, floats, or missing values.
    """
    try:
        return int(value)
    except Exception:
        return 0
    

def otsl_tokens(otsl: str) -> list[str]:
    """
    Split a normalized OTSL string into tokens.
    """
    return [
        token.strip()
        for token in str(otsl).replace(",", " ").split()
        if token.strip()
    ]


def detect_spans(otsl: str) -> bool:
    """
    Return True if OTSL contains merged/spanning-cell tokens.
    """
    tokens = set(otsl_tokens(otsl))
    return bool(tokens & SPAN_TOKENS)


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
    Create a stable sample ID for one FinTabNet sample.

    The dataset rows do not need to provide their own ID.
    We create one from split + index + table metadata + a short content hash.
    """
    digest_source = f"{split}|{source_index}|{rows}|{cols}|{gt_otsl}|{gt_html}"
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:12]

    return f"fintabnet_{split}_{source_index:06d}_{digest}"


def save_pil_image(
    *,
    image: Any,
    output_path: str | Path,
) -> None:
    """
    Save a HuggingFace/PIL image as RGB PNG.

    FinTabNet image field is expected to be a PIL image.
    We save it locally so it can later be converted into a PDF.
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


def materialize_fintabnet_sample(
    *,
    raw_sample: dict[str, Any],
    source_index: int,
    split: str = DEFAULT_SPLIT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> FinTabNetSample:
    """
    Convert one raw FinTabNet parquet row into a benchmark-ready sample.

    This creates:
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
        raise ValueError(f"FinTabNet sample index={source_index} has no image field.")

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


def iter_materialized_fintabnet_samples(
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    *,
    split: str = DEFAULT_SPLIT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
) -> list[FinTabNetSample]:
    """
    Materialize FinTabNet test samples from local raw parquet files.

    This is the one-time/dev setup path:
    raw parquet -> PNG files -> PDF files -> FinTabNetSample objects
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

    return samples


def save_manifest(
    samples: list[FinTabNetSample],
    manifest_path: str | Path,
) -> None:
    """
    Save materialized FinTabNet samples as a JSONL manifest.

    The manifest is small and reproducible.
    It can be committed to GitHub.
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
    """
    Load a FinTabNet JSONL manifest.

    This is the normal benchmark-runtime path:
    manifest -> rows used by runner
    """
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
    """
    Convert one JSONL manifest record back into a FinTabNetSample.

    This is used during normal benchmark runs after the manifest already exists.
    """
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
    """
    Load benchmark-ready FinTabNet samples from an existing manifest.

    This is the normal benchmark runtime path.
    """
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
    """
    Return the default local FinTabNet manifest path.
    """
    return Path(output_dir) / split / "manifest.jsonl"


def materialize_fintabnet_dataset(
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    *,
    split: str = DEFAULT_SPLIT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
    manifest_path: str | Path | None = None,
) -> list[FinTabNetSample]:
    """
    One-shot materialization helper.

    Raw parquet files
        -> PNG files
        -> PDF files
        -> manifest.jsonl
        -> list[FinTabNetSample]

    Use this during setup/dev, not every benchmark run.
    """
    samples = iter_materialized_fintabnet_samples(
        raw_data_dir=raw_data_dir,
        split=split,
        output_dir=output_dir,
        limit=limit,
    )

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