"""
OmniDocBench dataset loader and materializer.

OmniDocBench provides 1,651 annotated document pages (v1.6).
Each page has:
- a PDF in two forms: image-converted (pdfs/) and original digital (ori_pdfs/)
- rich layout annotations in OmniDocBench.json
- text content GT for NED scoring
- table HTML GT for TEDS scoring
- reading order GT for reading order NED scoring

This loader materializes samples from the local JSON + PDF folders and writes
a local runtime manifest for runner.py.

No extraction happens here.
No scoring happens here.
Scoring is handled by OmniDocBench's own eval harness via a converter.

Two input modes:
    digital  → ori_pdfs/   original source PDFs with text layers
                            tests native PDF extraction path
                            scores NOT directly comparable to official leaderboard
                            (original PDFs lack visual masking on 390 pages)

    scanned  → pdfs/        image-converted PDFs, no text layer
                            forces OCR path on every engine
                            scores directly comparable to official leaderboard

OmniDocBench.json (GT) + PDF folders
        ↓
omnidocbench.py reads JSON
        ↓
resolves pdf_path from image_path filename stem
        ↓
filters layout_dets by category
        ↓
extracts reading order (ignore=False, sorted by order field)
        ↓
writes manifest.jsonl with both digital and scanned PDF paths
        ↓
runner.py later reads manifest
        ↓
runner sends pdf_path to adapter
        ↓
converter converts DocumentResult → OmniDocBench prediction JSON
        ↓
OmniDocBench eval harness scores NED + TEDS + reading order

Coordinate space note:
    OmniDocBench GT poly coordinates are in image pixel space
    (page_info.width × page_info.height, e.g. 1653×2339).
    poly is a quadrilateral: [x1,y1, x2,y2, x3,y3, x4,y4] — 8 floats.
    poly_to_xyxy() converts to axis-aligned xyxy for bbox matching.
    The converter (not this loader) handles coordinate space alignment
    between adapter output and GT before calling the eval harness.

Ignore flag note:
    layout_dets entries with ignore=True participate in bbox matching
    but are excluded from metric calculations.
    This loader preserves ignore=True entries — do not drop them.
    The eval harness handles ignore logic internally.

Mask note:
    368 pages have abandon areas (headers/footers with special graphics).
    22 pages have unparseable areas (tables containing images).
    These are marked with ignore=True in layout_dets.
    ori_pdfs/ does not have visual masking applied to these regions.
    pdfs/ (image-converted) does have masking applied.
    This is the key difference between digital and scanned modes.

Download note:
    Full dataset is ~1.31GB on HuggingFace (opendatalab/OmniDocBench).
    images/ folder is skipped during download — not needed for PDF evaluation.
    Effective download size is ~865MB (JSON + pdfs/ + ori_pdfs/).
    Download is one-time only. Subsequent runs use local manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DATASET_SLUG = "opendatalab/OmniDocBench"
EXPECTED_COUNT = 1651

# env var pattern — same as doclaynet.py and fintabnet.py
# On Kaggle:  export OMNIDOCBENCH_RAW_DIR=/kaggle/input/omnidocbench
# On RunPod:  export OMNIDOCBENCH_RAW_DIR=/workspace/omnidocbench_raw
DEFAULT_RAW_DATA_DIR = Path(os.getenv("OMNIDOCBENCH_RAW_DIR", "data/omnidocbench_raw"))
DEFAULT_OUTPUT_DIR   = Path(os.getenv("OMNIDOCBENCH_OUTPUT_DIR", "data/omnidocbench"))

# Two input modes — controls which PDF folder is used
INPUT_TYPE_DIGITAL = "digital"   # ori_pdfs/ — original source PDFs with text layers
INPUT_TYPE_SCANNED = "scanned"   # pdfs/     — image-converted PDFs, no text layer
DEFAULT_INPUT_TYPE = INPUT_TYPE_DIGITAL

# GT JSON filename — confirmed present in the HuggingFace repo
GT_JSON_FILENAME = "OmniDocBench.json"

# Category types confirmed from real data inspection.
# Full list from: sorted(set(det["category_type"] for page in data for det in page["layout_dets"]))
# Score these for text NED:
TEXT_CATEGORIES = {
    "text_block",
    "title",
    "list_group",
    "code_txt",
    "code_txt_caption",
    "reference",
    "page_footnote",
    "header",
    "footer",
    "page_number",
    "equation_caption",
    "equation_explanation",
    "equation_semantic",
    "figure_caption",
    "figure_footnote",
    "table_caption",
    "table_footnote",
}

# Score these for structure:
TABLE_CATEGORIES = {"table"}
FORMULA_CATEGORIES = {"equation_isolated"}    # bbox detection only — CDM excluded per spec
FIGURE_CATEGORIES = {"figure"}

# Skip entirely — mask/ignore categories
# The ignore flag in layout_dets handles per-box ignoring.
# These category types are always non-scoring.
MASK_CATEGORIES = {
    "abandon",
    "algorithm_mask",
    "chart_mask",
    "need_mask",
    "organic_chemical_formula_mask",
    "table_mask",
    "text_mask",
    "unknown_mask",
}

# All known category types from real data.
# Raise if an unknown type appears — dataset schema change must be caught loudly.
ALL_KNOWN_CATEGORIES = (
    TEXT_CATEGORIES
    | TABLE_CATEGORIES
    | FORMULA_CATEGORIES
    | FIGURE_CATEGORIES
    | MASK_CATEGORIES
)


@dataclass(frozen=True, slots=True)
class OmniDocBenchSample:
    """
    One benchmark-ready OmniDocBench page sample.

    Stores paths for both input modes so runner.py can pick
    digital or scanned at run time without re-materializing.

    GT fields are preserved exactly as-is from OmniDocBench.json.
    The converter (not this loader) transforms them into the
    prediction comparison format expected by the eval harness.
    """

    sample_id: str
    source_index: int
    image_filename: str          # "page-xxx.png" — join key between JSON and PDF folders

    pdf_path_digital: Path       # ori_pdfs/page-xxx.pdf — original source PDF
    pdf_path_scanned: Path       # pdfs/page-xxx.pdf     — image-converted PDF

    # GT fields — preserved faithfully from OmniDocBench.json
    gt_layout_dets: list[dict]   # full raw layout_dets — all categories, all ignore flags
    gt_text_blocks: list[dict]   # filtered: TEXT_CATEGORIES only, ignore=False
    gt_tables: list[dict]        # filtered: TABLE_CATEGORIES only, ignore=False
    gt_reading_order: list[dict] # ignore=False entries sorted by order field

    # GT bboxes converted to xyxy for layout scorer
    gt_bboxes_xyxy: list[list[float]]      # parallel to gt_layout_dets
    gt_bboxes_raw_poly: list[list[float]]  # original 8-float quad polys, preserved for audit

    page_width: float            # pixels — from page_info.width
    page_height: float           # pixels — from page_info.height

    # Page-level attributes from page_info.page_attribute
    data_source: str             # "book", "financial_reports", "paper", etc.
    language: str                # "english", "chinese", "mixed"
    layout: str                  # "single_column", "double_column", "complex", etc.
    special_issues: list[str]    # ["fuzzy_scan", "watermark", "colorful_background"] etc.

    has_table: bool
    has_formula: bool
    has_figure: bool

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_manifest_record(self) -> dict[str, Any]:
        # Explicit field-by-field — no asdict().
        # Path fields serialized as str.
        # Adding a new field here can never create a silent JSON serialization bug.
        return {
            "sample_id": self.sample_id,
            "source_index": self.source_index,
            "image_filename": self.image_filename,
            "pdf_path_digital": str(self.pdf_path_digital),
            "pdf_path_scanned": str(self.pdf_path_scanned),
            "gt_layout_dets": self.gt_layout_dets,
            "gt_text_blocks": self.gt_text_blocks,
            "gt_tables": self.gt_tables,
            "gt_reading_order": self.gt_reading_order,
            "gt_bboxes_xyxy": self.gt_bboxes_xyxy,
            "gt_bboxes_raw_poly": self.gt_bboxes_raw_poly,
            "page_width": self.page_width,
            "page_height": self.page_height,
            "data_source": self.data_source,
            "language": self.language,
            "layout": self.layout,
            "special_issues": self.special_issues,
            "has_table": self.has_table,
            "has_formula": self.has_formula,
            "has_figure": self.has_figure,
            "metadata": self.metadata,
        }

    def pdf_path(self, input_type: str = DEFAULT_INPUT_TYPE) -> Path:
        """
        Return the correct PDF path for the given input type.

        input_type="digital" → ori_pdfs/ (original source PDF with text layer)
        input_type="scanned" → pdfs/ (image-converted PDF, OCR path)
        """
        if input_type == INPUT_TYPE_DIGITAL:
            return self.pdf_path_digital
        if input_type == INPUT_TYPE_SCANNED:
            return self.pdf_path_scanned
        raise ValueError(
            f"Unknown input_type={input_type!r}. "
            f"Use '{INPUT_TYPE_DIGITAL}' or '{INPUT_TYPE_SCANNED}'."
        )


def make_sample_id(
    *,
    source_index: int,
    image_filename: str,
) -> str:
    """
    Deterministic sample ID — no public manifest needed.

    Two people running prepare_omnidocbench() independently on the same
    local JSON will get identical sample IDs.

    Not input_type-specific — the sample is the same page regardless of
    which PDF folder is used. input_type is a runner concern, not a sample concern.
    """
    digest_source = f"{source_index}|{image_filename}"
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:12]

    return f"omnidocbench_{source_index:06d}_{digest}"


def poly_to_xyxy(poly: list[float]) -> list[float]:
    """
    Convert OmniDocBench quadrilateral poly to axis-aligned xyxy bbox.

    OmniDocBench poly format:
        [x1,y1, x2,y2, x3,y3, x4,y4]  — 8 floats, clockwise quad

    Output:
        [x0, y0, x1, y1]  — axis-aligned bounding box

    For axis-aligned pages this is lossless.
    For rotated or skewed boxes the xyxy bbox is the tight enclosure.
    The original poly is preserved in gt_bboxes_raw_poly for audit.
    """
    if len(poly) != 8:
        raise ValueError(f"OmniDocBench poly must have 8 floats, got {len(poly)}: {poly!r}")

    xs = poly[0::2]
    ys = poly[1::2]

    return [min(xs), min(ys), max(xs), max(ys)]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _validate_category_types(dets: list[dict], source_index: int) -> None:
    """
    Raise if any layout_det has an unknown category_type.

    Unknown categories mean the dataset schema changed or the JSON is corrupt.
    Both cases must fail loudly — silently ignoring them would corrupt GT.

    Same pattern as doclaynet.py _validate_category_ids().
    """
    unknown = sorted(
        set(det.get("category_type", "") for det in dets)
        - ALL_KNOWN_CATEGORIES
        - {""}
    )

    if unknown:
        raise ValueError(
            f"Unknown OmniDocBench category_types at source_index={source_index}: "
            f"{unknown}. Update ALL_KNOWN_CATEGORIES if the dataset schema changed."
        )


def _filter_text_blocks(dets: list[dict]) -> list[dict]:
    """
    Return layout_dets entries that are text-content categories with ignore=False.

    These are the GT blocks used for text NED scoring.
    Entries with ignore=True are excluded — they participate in bbox matching
    inside the eval harness but are not scored.
    """
    return [
        det for det in dets
        if det.get("category_type") in TEXT_CATEGORIES
        and not det.get("ignore", False)
    ]


def _filter_tables(dets: list[dict]) -> list[dict]:
    """
    Return layout_dets entries that are table categories with ignore=False.

    These are the GT tables used for TEDS scoring.
    """
    return [
        det for det in dets
        if det.get("category_type") in TABLE_CATEGORIES
        and not det.get("ignore", False)
    ]


def _extract_reading_order(dets: list[dict]) -> list[dict]:
    """
    Return layout_dets entries sorted by reading order, ignore=False only.

    OmniDocBench reading order is stored in the order field per layout_det.
    Entries without an order field or with ignore=True are excluded.

    The eval harness uses this list to compute reading order NED.
    """
    ordered = [
        det for det in dets
        if det.get("order") is not None
        and not det.get("ignore", False)
    ]

    return sorted(ordered, key=lambda d: int(d["order"]))


def _pdf_stem_from_image_path(image_path: str) -> str:
    """
    Extract the filename stem from page_info.image_path.

    page_info.image_path = "page-d1561665-5359-42fe-920c-d6e3bff81953.png"
    stem                  = "page-d1561665-5359-42fe-920c-d6e3bff81953"

    The PDF files use the same stem:
        pdfs/page-d1561665-5359-42fe-920c-d6e3bff81953.pdf
        ori_pdfs/page-d1561665-5359-42fe-920c-d6e3bff81953.pdf
    """
    return Path(image_path).stem


def materialize_omnidocbench_sample(
    *,
    raw_record: dict[str, Any],
    source_index: int,
    raw_data_dir: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> OmniDocBenchSample:
    """
    Convert one raw OmniDocBench JSON record into a benchmark-ready sample.

    Steps:
    - read page_info, layout_dets, extra
    - validate all category_types (raise on unknown — never skip silently)
    - resolve pdf_path_digital and pdf_path_scanned from image_filename stem
    - convert poly → xyxy bboxes
    - filter text blocks and tables
    - extract reading order
    - build and return OmniDocBenchSample

    No PDF is written to disk here — PDFs already exist in raw_data_dir.
    This loader only builds the manifest record pointing at existing PDFs.
    """
    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)

    page_info = raw_record.get("page_info") or {}
    layout_dets = raw_record.get("layout_dets") or []

    image_path_str = str(page_info.get("image_path", ""))
    if not image_path_str:
        raise ValueError(
            f"OmniDocBench record index={source_index} has no page_info.image_path."
        )

    stem = _pdf_stem_from_image_path(image_path_str)
    image_filename = Path(image_path_str).name

    # Resolve both PDF paths from the same filename stem
    pdf_path_digital = raw_data_dir / "ori_pdfs" / f"{stem}.pdf"
    pdf_path_scanned = raw_data_dir / "pdfs" / f"{stem}.pdf"

    # Only validate existence after confirming folders exist.
    # If folders are present but specific file is missing, that is a real error.
    # If folders don't exist yet, prepare_omnidocbench() handles download first.
    if (raw_data_dir / "ori_pdfs").exists() and not pdf_path_digital.exists():
        raise FileNotFoundError(
            f"Missing OmniDocBench digital PDF for source_index={source_index}: "
            f"{pdf_path_digital}"
        )

    if (raw_data_dir / "pdfs").exists() and not pdf_path_scanned.exists():
        raise FileNotFoundError(
            f"Missing OmniDocBench scanned PDF for source_index={source_index}: "
            f"{pdf_path_scanned}"
        )

    # Validate category types — raise on unknown, never skip
    _validate_category_types(layout_dets, source_index)

    sample_id = make_sample_id(
        source_index=source_index,
        image_filename=image_filename,
    )

    # Convert all polys to xyxy — preserve raw polys for audit
    gt_bboxes_raw_poly: list[list[float]] = []
    gt_bboxes_xyxy: list[list[float]] = []

    for det in layout_dets:
        poly = det.get("poly") or []

        if len(poly) == 8:
            gt_bboxes_raw_poly.append([float(v) for v in poly])
            gt_bboxes_xyxy.append(poly_to_xyxy([float(v) for v in poly]))
        else:
            # Malformed poly — preserve zeros so indices stay aligned
            gt_bboxes_raw_poly.append([0.0] * 8)
            gt_bboxes_xyxy.append([0.0, 0.0, 0.0, 0.0])

            print(
                f"WARNING: Malformed poly (len={len(poly)}) at "
                f"source_index={source_index}, anno_id={det.get('anno_id')} — "
                f"using zero bbox."
            )

    # Page dimensions — from page_info, confirmed present in real data
    page_width  = safe_float(page_info.get("width"))
    page_height = safe_float(page_info.get("height"))

    if page_width == 0.0 and page_height == 0.0:
        print(
            f"WARNING: page dimensions are 0.0 for source_index={source_index}. "
            f"page_info keys seen: {sorted(page_info.keys())}"
        )

    # Page attributes — from page_info.page_attribute
    page_attr    = page_info.get("page_attribute") or {}
    data_source  = str(page_attr.get("data_source", "unknown"))
    language     = str(page_attr.get("language", "unknown"))
    layout       = str(page_attr.get("layout", "unknown"))
    special_issues = list(page_attr.get("special_issue") or [])

    # Category presence flags
    cats = {det.get("category_type") for det in layout_dets}
    has_table   = bool(cats & TABLE_CATEGORIES)
    has_formula = bool(cats & FORMULA_CATEGORIES)
    has_figure  = bool(cats & FIGURE_CATEGORIES)

    metadata = {
        "dataset_slug": DATASET_SLUG,
        "source_index": source_index,
        "image_filename": image_filename,
        "source_format": "pdf_page",
        "pdf_path_digital": str(pdf_path_digital),
        "pdf_path_scanned": str(pdf_path_scanned),
        "field_names_seen": sorted(raw_record.keys()),
    }

    return OmniDocBenchSample(
        sample_id=sample_id,
        source_index=source_index,
        image_filename=image_filename,
        pdf_path_digital=pdf_path_digital,
        pdf_path_scanned=pdf_path_scanned,
        gt_layout_dets=layout_dets,
        gt_text_blocks=_filter_text_blocks(layout_dets),
        gt_tables=_filter_tables(layout_dets),
        gt_reading_order=_extract_reading_order(layout_dets),
        gt_bboxes_xyxy=gt_bboxes_xyxy,
        gt_bboxes_raw_poly=gt_bboxes_raw_poly,
        page_width=page_width,
        page_height=page_height,
        data_source=data_source,
        language=language,
        layout=layout,
        special_issues=special_issues,
        has_table=has_table,
        has_formula=has_formula,
        has_figure=has_figure,
        metadata=metadata,
    )


def save_manifest(
    samples: list[OmniDocBenchSample],
    manifest_path: str | Path,
) -> None:
    """
    Save the local runtime OmniDocBench manifest.

    Local artifact only — gitignored.
    Reproducibility anchor is the run summary JSON, not this file.

    One manifest exists regardless of input_type — OmniDocBenchSample stores
    both pdf_path_digital and pdf_path_scanned. The runner picks which path
    to use at run time via sample.pdf_path(input_type).
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


def sample_from_manifest_record(record: dict[str, Any]) -> OmniDocBenchSample:
    """
    Restore one OmniDocBenchSample from a manifest JSONL record.

    Used during normal benchmark runs after the manifest already exists.
    """
    return OmniDocBenchSample(
        sample_id=str(record["sample_id"]),
        source_index=int(record["source_index"]),
        image_filename=str(record["image_filename"]),
        pdf_path_digital=Path(record["pdf_path_digital"]),
        pdf_path_scanned=Path(record["pdf_path_scanned"]),
        gt_layout_dets=list(record.get("gt_layout_dets") or []),
        gt_text_blocks=list(record.get("gt_text_blocks") or []),
        gt_tables=list(record.get("gt_tables") or []),
        gt_reading_order=list(record.get("gt_reading_order") or []),
        gt_bboxes_xyxy=[
            [float(v) for v in bbox]
            for bbox in record.get("gt_bboxes_xyxy") or []
        ],
        gt_bboxes_raw_poly=[
            [float(v) for v in poly]
            for poly in record.get("gt_bboxes_raw_poly") or []
        ],
        page_width=safe_float(record.get("page_width")),
        page_height=safe_float(record.get("page_height")),
        data_source=str(record.get("data_source", "unknown")),
        language=str(record.get("language", "unknown")),
        layout=str(record.get("layout", "unknown")),
        special_issues=list(record.get("special_issues") or []),
        has_table=bool(record.get("has_table", False)),
        has_formula=bool(record.get("has_formula", False)),
        has_figure=bool(record.get("has_figure", False)),
        metadata=dict(record.get("metadata") or {}),
    )


def iter_omnidocbench_samples_from_manifest(
    manifest_path: str | Path,
    *,
    limit: int | None = None,
) -> list[OmniDocBenchSample]:
    """
    Load benchmark-ready OmniDocBench samples from an existing manifest.

    This is the normal benchmark runtime path —
    manifest already exists, runner reads it directly.
    """
    records = load_manifest(
        manifest_path=manifest_path,
        limit=limit,
    )

    return [sample_from_manifest_record(record) for record in records]


def default_manifest_path(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """
    Return the default local OmniDocBench manifest path.

    One manifest covers both input types — OmniDocBenchSample stores both
    pdf_path_digital and pdf_path_scanned. No separate manifest per input_type.
    """
    return Path(output_dir) / "manifest.jsonl"


def _manifest_is_sufficient(
    manifest_path: str | Path,
    limit: int | None,
    expected_full_count: int = EXPECTED_COUNT,
) -> bool:
    """
    Return True if the manifest has enough rows for this run.

    limit=None  → need full expected count (1651)
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


def _gt_json_path(raw_data_dir: str | Path) -> Path:
    """
    Return the expected path for OmniDocBench.json.
    """
    return Path(raw_data_dir) / GT_JSON_FILENAME


def _pdf_folders_exist(raw_data_dir: str | Path) -> bool:
    """
    Return True if both pdfs/ and ori_pdfs/ folders exist and are non-empty.

    Both folders are required — the loader needs both input types.
    """
    raw_data_dir = Path(raw_data_dir)

    pdfs_dir     = raw_data_dir / "pdfs"
    ori_pdfs_dir = raw_data_dir / "ori_pdfs"

    if not pdfs_dir.exists() or not ori_pdfs_dir.exists():
        return False

    # Check at least one PDF exists in each folder
    has_pdfs     = any(pdfs_dir.glob("*.pdf"))
    has_ori_pdfs = any(ori_pdfs_dir.glob("*.pdf"))

    return has_pdfs and has_ori_pdfs


def _download_omnidocbench(
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
) -> Path:
    """
    Download OmniDocBench from HuggingFace into a local folder.

    Downloads:
        OmniDocBench.json   ← GT annotations (65MB)
        pdfs/               ← image-converted PDFs, scanned path (~400MB)
        ori_pdfs/           ← original source PDFs, digital path (~400MB)

    Skips:
        images/             ← not needed for PDF evaluation (~400MB saved)

    Total download: ~865MB. One-time only.

    Called by prepare_omnidocbench() when GT JSON or PDF folders are missing.
    prepare_omnidocbench() calls materialize_omnidocbench_dataset() directly,
    which calls load_gt_json() which raises a clear error if the file is missing.
    _download_omnidocbench() is the explicit opt-in for pre-downloading.
    """
    from huggingface_hub import snapshot_download

    raw_data_dir = Path(raw_data_dir)
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading OmniDocBench to {raw_data_dir} ...")
    print("Skipping images/ folder — not needed for PDF evaluation.")

    snapshot_download(
        repo_id=DATASET_SLUG,
        repo_type="dataset",
        local_dir=str(raw_data_dir),
        # Skip images — saves ~400MB, not needed for PDF evaluation
        ignore_patterns=["images/*", "*.jpg", "*.jpeg", "*.png"],
    )

    print("Download complete.")

    return raw_data_dir


def load_gt_json(raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR) -> list[dict]:
    """
    Load OmniDocBench.json from the local raw data directory.

    Raises a clear error if the file is missing — tells the user exactly
    how to fix it rather than producing a confusing Python traceback.
    """
    gt_path = _gt_json_path(raw_data_dir)

    if not gt_path.exists():
        raise FileNotFoundError(
            f"OmniDocBench.json not found at {gt_path}. "
            "Run _download_omnidocbench() to download the dataset, "
            "or set OMNIDOCBENCH_RAW_DIR to the folder containing OmniDocBench.json."
        )

    with gt_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"OmniDocBench.json is not a list. Got {type(data).__name__}. "
            "The file may be corrupt or from an incompatible version."
        )

    return data


def materialize_omnidocbench_dataset(
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
    manifest_path: str | Path | None = None,
) -> list[OmniDocBenchSample]:
    """
    Materialize OmniDocBench samples from local JSON + PDF folders.

    Unlike DocLayNet and FinTabNet, no streaming or HF download happens here.
    The GT JSON is read entirely into memory (65MB — acceptable).
    PDFs already exist on disk in raw_data_dir/pdfs/ and raw_data_dir/ori_pdfs/.

    Raises FileNotFoundError if OmniDocBench.json is missing — user must
    call _download_omnidocbench() first or set OMNIDOCBENCH_RAW_DIR correctly.
    """
    raw_data_dir = Path(raw_data_dir)

    # Validate PDF folders exist before iterating 1651 records
    if not _pdf_folders_exist(raw_data_dir):
        raise FileNotFoundError(
            f"PDF folders not found at {raw_data_dir}. "
            "Expected pdfs/ and ori_pdfs/ subfolders. "
            "Run _download_omnidocbench() to download the dataset."
        )

    gt_records = load_gt_json(raw_data_dir)

    samples: list[OmniDocBenchSample] = []

    for source_index, raw_record in enumerate(gt_records):
        if limit is not None and len(samples) >= limit:
            break

        sample = materialize_omnidocbench_sample(
            raw_record=raw_record,
            source_index=source_index,
            raw_data_dir=raw_data_dir,
            output_dir=output_dir,
        )

        samples.append(sample)

    if manifest_path is None:
        manifest_path = default_manifest_path(output_dir=output_dir)

    save_manifest(
        samples=samples,
        manifest_path=manifest_path,
    )

    return samples


def prepare_omnidocbench(
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
    manifest_path: str | Path | None = None,
) -> Path:
    """
    Top-level entry point. This is what runner.py calls.

    Flow:
        1. Manifest exists and has enough rows → return immediately
        2. GT JSON + PDF folders exist locally → materialize from them
        3. Nothing exists → download from HuggingFace → materialize

    Unlike DocLayNet and FinTabNet, OmniDocBench does not use HF streaming.
    The GT JSON is read directly. PDFs are already on disk after download.

    Returns manifest path ready for iter_omnidocbench_samples_from_manifest().
    """
    if manifest_path is None:
        manifest_path = default_manifest_path(output_dir=output_dir)

    manifest_path = Path(manifest_path)

    # Step 1 — manifest already sufficient
    if _manifest_is_sufficient(manifest_path=manifest_path, limit=limit):
        return manifest_path

    raw_data_dir = Path(raw_data_dir)

    # Step 2 — download if GT JSON or PDF folders are missing
    gt_missing   = not _gt_json_path(raw_data_dir).exists()
    pdfs_missing = not _pdf_folders_exist(raw_data_dir)

    if gt_missing or pdfs_missing:
        _download_omnidocbench(raw_data_dir=raw_data_dir)

    # Step 3 — materialize from local JSON + PDF folders
    materialize_omnidocbench_dataset(
        raw_data_dir=raw_data_dir,
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
        print(f"WARNING: Could not fetch HF commit hash for {slug}: {exc}")
        return "unknown"