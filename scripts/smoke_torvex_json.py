import argparse
import json
from pathlib import Path

from torvex_bench.adapters.torvex_extract_adapter import TorvexExtractAdapter
from torvex_bench.normalizer import normalize_document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("--out", default="results/smoke/torvex_normalized.json")
    args = parser.parse_args()

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    adapter = TorvexExtractAdapter()
    document = adapter.extract_document(args.pdf_path)
    normalized = normalize_document(document)

    output_path.write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"saved: {output_path}")
    print(f"pages: {len(normalized['pages'])}")
    print(f"errors: {len(normalized.get('errors', []))}")


if __name__ == "__main__":
    main()