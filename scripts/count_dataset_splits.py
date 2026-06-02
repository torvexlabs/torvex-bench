from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import load_dataset

import json

RESULTS_DIR = Path("results")
RESULTS_PATH = RESULTS_DIR / "dataset_counts.json"

def short_value(value: Any, max_len: int = 120) -> str:
    """
    Return a short printable preview of any dataset field value.
    """
    if value is None:
        return "None"

    text = repr(value).replace("\n", "\\n")

    if len(text) > max_len:
        return text[:max_len] + "..."

    return text


def peek_dataset(
    dataset_name: str,
    split: str = "test",
    streaming: bool = True,
):
    """
    Safely open one dataset split and print the first sample's fields.
    """
    ds = load_dataset(
        dataset_name,
        split=split,
        streaming=streaming,
    )

    if streaming:
        sample = next(iter(ds))
    else:
        sample = ds[0]

    print(f"\n=== {dataset_name} / {split} ===")
    print(f"streaming: {streaming}")
    print(f"fields: {list(sample.keys())}")
    print("\nfield preview:")

    for key, value in sample.items():
        print(f"  - {key}: {type(value).__name__} = {short_value(value)}")

    return ds, sample


def count_doclaynet(limit: int | None = 20) -> dict[str, Any]:
    """
    Count basic DocLayNet information.

    Beginner-safe default:
    - limit=20 counts only first 20 samples.
    - Later, limit=None will count the full test split.
    """
    dataset_name = "docling-project/DocLayNet-v1.2"

    ds, first_sample = peek_dataset(
        dataset_name,
        split="test",
        streaming=True,
    )

    total = 0
    pages_with_tables = 0
    pages_with_formulas = 0
    pdf_available = 0

    for sample in ds:
        if limit is not None and total >= limit:
            break

        total += 1

        if sample.get("pdf") or sample.get("pdf_bytes"):
            pdf_available += 1

        category_ids = sample.get("category_id") or []

        if 9 in category_ids:
            pages_with_tables += 1

        if 3 in category_ids:
            pages_with_formulas += 1

    result = {
        "dataset": dataset_name,
        "split": "test",
        "limit": limit,
        "counted_samples": total,
        "pdf_available": pdf_available,
        "pages_with_tables": pages_with_tables,
        "pages_with_formulas": pages_with_formulas,
        "fields": list(first_sample.keys()),
    }

    print("\nDocLayNet count result:")
    print(result)

    return result


def count_fintabnet(limit: int | None = 20) -> dict[str, Any]:
    """
    Count basic FinTabNet OTSL information.

    Beginner-safe default:
    - limit=20 checks only first 20 table crops.
    - Later, limit=1000 for the benchmark subset.
    """
    dataset_name = "docling-project/FinTabNet_OTSL"

    ds, first_sample = peek_dataset(
        dataset_name,
        split="test",
        streaming=True,
    )

    total = 0
    images_available = 0
    html_available = 0
    otsl_available = 0

    for sample in ds:
        if limit is not None and total >= limit:
            break

        total += 1

        if sample.get("image") is not None:
            images_available += 1

        if sample.get("html") or sample.get("table_html"):
            html_available += 1

        if sample.get("otsl") or sample.get("tokens"):
            otsl_available += 1

    result = {
        "dataset": dataset_name,
        "split": "test",
        "limit": limit,
        "counted_samples": total,
        "images_available": images_available,
        "html_available": html_available,
        "otsl_available": otsl_available,
        "fields": list(first_sample.keys()),
    }

    print("\nFinTabNet count result:")
    print(result)

    return result


def count_omnidocbench(limit: int | None = 20) -> dict[str, Any]:
    """
    Count basic OmniDocBench information.

    Beginner-safe default:
    - limit=20 checks only first 20 samples.
    - Later, limit=None can count the full test split.
    """
    dataset_name = "opendatalab/OmniDocBench"

    ds, first_sample = peek_dataset(
        dataset_name,
        split="train",
        streaming=True,
    )

    total = 0
    pages_with_tables = 0
    pages_with_formulas = 0
    possible_scanned_pages = 0

    for sample in ds:
        if limit is not None and total >= limit:
            break

        total += 1

        sample_text = repr(sample).lower()

        if "table" in sample_text:
            pages_with_tables += 1

        if "formula" in sample_text:
            pages_with_formulas += 1

        if "scan" in sample_text or "scanned" in sample_text:
            possible_scanned_pages += 1

    result = {
        "dataset": dataset_name,
        "split": "train",
        "limit": limit,
        "counted_samples": total,
        "pages_with_tables": pages_with_tables,
        "pages_with_formulas": pages_with_formulas,
        "possible_scanned_pages": possible_scanned_pages,
        "fields": list(first_sample.keys()),
    }

    print("\nOmniDocBench count result:")
    print(result)

    return result


def count_olmocr() -> dict[str, Any]:
    """
    Record olmOCR-bench basic metadata.

    We do not download/load the full dataset here yet.
    """
    dataset_name = "allenai/olmOCR-bench"

    result = {
        "dataset": dataset_name,
        "split": "benchmark",
        "pdfs": 1403,
        "unit_tests": 7010,
        "metric": "unit_test_pass_rate",
        "note": "Use unit test pass rate only. Do not use CER for olmOCR-bench.",
    }

    print("\nolmOCR-bench count result:")
    print(result)

    return result



def main() -> None:
    """
    Run all dataset counting functions and save results to JSON.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "doclaynet": count_doclaynet(limit=20),
        "fintabnet": count_fintabnet(limit=20),
        "omnidocbench": count_omnidocbench(limit=20),
        "olmocr": count_olmocr(),
    }

    RESULTS_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved dataset counts to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()