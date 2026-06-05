from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from torvex_bench.adapters.base import ExtractionAdapter
from torvex_bench.runner import count_successes, run_samples


SUPPORTED_DATASETS = {
    "fintabnet",
    "doclaynet",
    "omnidocbench",
    "olmocr",
}

SUPPORTED_ENGINES = {
    "torvex",
    "docling",
    "ppstructure",
}

DEFAULT_RUN_OUTPUT_DIR   = Path("results/raw")
DEFAULT_SCORE_OUTPUT_DIR = Path("results/scores")


def positive_int_or_none(value: str | None) -> int | None:
    if value is None:
        return None

    parsed = int(value)

    if parsed <= 0:
        raise argparse.ArgumentTypeError("limit must be a positive integer")

    return parsed


def _optional_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """
    Build optional loader kwargs from CLI args.

    Only include raw_data_dir and dataset_output_dir when explicitly set.
    Passing None would bypass loader defaults and crash with TypeError.
    """
    kwargs: dict[str, Any] = {}

    if getattr(args, "raw_data_dir", None) is not None:
        kwargs["raw_data_dir"] = args.raw_data_dir

    if getattr(args, "dataset_output_dir", None) is not None:
        kwargs["output_dir"] = args.dataset_output_dir

    return kwargs


def make_adapter(engine: str) -> ExtractionAdapter:
    """
    Instantiate one benchmark adapter.

    Only Torvex is production-ready at this stage.
    Docling and PPStructure are reserved for the full benchmark matrix.
    """
    engine = engine.strip().lower()

    if engine == "torvex":
        from torvex_bench.adapters.torvex_extract_adapter import TorvexExtractAdapter
        return TorvexExtractAdapter()

    if engine == "docling":
        try:
            from torvex_bench.adapters.docling_adapter import DoclingAdapter
        except Exception as exc:
            raise RuntimeError(
                "Docling adapter is not available yet. "
                "Implement src/torvex_bench/adapters/docling_adapter.py first."
            ) from exc
        return DoclingAdapter()

    if engine == "ppstructure":
        try:
            from torvex_bench.adapters.ppstructure_adapter import PPStructureAdapter
        except Exception as exc:
            raise RuntimeError(
                "PPStructure adapter is not available yet. "
                "Implement src/torvex_bench/adapters/ppstructure_adapter.py first."
            ) from exc
        return PPStructureAdapter()

    raise ValueError(f"Unknown engine: {engine!r}")


# ── Dataset loaders ───────────────────────────────────────────────────────────

def prepare_and_load_fintabnet(args: argparse.Namespace) -> list[Any]:
    from torvex_bench.datasets.fintabnet import (
        iter_fintabnet_samples_from_manifest,
        prepare_fintabnet,
    )

    kwargs = _optional_kwargs(args)
    manifest_path = prepare_fintabnet(
        split=args.split,
        limit=args.limit,
        **kwargs,
    )

    return iter_fintabnet_samples_from_manifest(
        manifest_path=manifest_path,
        limit=args.limit,
    )


def prepare_and_load_doclaynet(args: argparse.Namespace) -> list[Any]:
    from torvex_bench.datasets.doclaynet import (
        iter_doclaynet_samples_from_manifest,
        prepare_doclaynet,
    )

    kwargs = _optional_kwargs(args)
    manifest_path = prepare_doclaynet(
        split=args.split,
        limit=args.limit,
        **kwargs,
    )

    return iter_doclaynet_samples_from_manifest(
        manifest_path=manifest_path,
        limit=args.limit,
    )


def prepare_and_load_omnidocbench(args: argparse.Namespace) -> list[Any]:
    from torvex_bench.datasets.omnidocbench import (
        iter_omnidocbench_samples_from_manifest,
        prepare_omnidocbench,
    )

    kwargs = _optional_kwargs(args)
    manifest_path = prepare_omnidocbench(
        limit=args.limit,
        **kwargs,
    )

    return iter_omnidocbench_samples_from_manifest(
        manifest_path=manifest_path,
        limit=args.limit,
    )


def prepare_and_load_olmocr(args: argparse.Namespace) -> list[Any]:
    from torvex_bench.datasets.olmocr_bench import (
        iter_olmocr_samples_from_manifest,
        prepare_olmocr_bench,
    )

    kwargs = _optional_kwargs(args)
    manifest_path = prepare_olmocr_bench(
        subset=args.olmocr_subset,
        limit=args.limit,
        download_pdfs=not args.no_download_pdfs,
        **kwargs,
    )

    return iter_olmocr_samples_from_manifest(
        manifest_path=manifest_path,
        limit=args.limit,
    )


DATASET_LOADERS: dict[str, Callable[[argparse.Namespace], list[Any]]] = {
    "fintabnet":    prepare_and_load_fintabnet,
    "doclaynet":    prepare_and_load_doclaynet,
    "omnidocbench": prepare_and_load_omnidocbench,
    "olmocr":       prepare_and_load_olmocr,
}


# ── Scorers ───────────────────────────────────────────────────────────────────

def _engine_label_from_name(engine: str) -> str:
    """
    Map CLI engine name to adapter .name attribute.
    Keeps output paths consistent with run command.
    """
    return {
        "torvex":      "torvex_extract",
        "docling":     "docling",
        "ppstructure": "ppstructure",
    }.get(engine.strip().lower(), engine.strip().lower())


def _normalized_path(
    run_output_dir: Path,
    dataset: str,
    engine_label: str,
    sample_id: str,
) -> Path:
    return run_output_dir / dataset / engine_label / "normalized" / f"{sample_id}.json"


def _load_normalized(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def score_fintabnet(
    *,
    samples: list[Any],
    engine_label: str,
    run_output_dir: Path,
    score_output_dir: Path,
    dataset: str = "fintabnet",
    engine: str,
    scored_at: str,
) -> int:
    """
    Score FinTabNet samples against saved normalized outputs.

    Reads normalized JSONs written by runner.
    Scores each against GT from manifest using TEDS scorer.
    Saves per-sample scores JSONL + summary JSON.
    Returns exit code: 0 = all scored, 1 = some missing/errors.
    """
    from torvex_bench.scorers.table_structure import (
        save_table_structure_scores_jsonl,
        score_fintabnet_sample,
        summarize_table_structure_scores,
    )
    from torvex_bench.datasets.fintabnet import get_hf_dataset_commit

    scores = []
    missing = 0

    for sample in samples:
        norm_path = _normalized_path(
            run_output_dir, dataset, engine_label, sample.sample_id
        )

        normalized = _load_normalized(norm_path)

        if normalized is None:
            print(
                f"  [skip] {sample.sample_id} — normalized output not found at {norm_path}"
            )
            missing += 1
            continue

        score = score_fintabnet_sample(
            sample=sample,
            document_result=normalized,
        )

        scores.append(score)

        status = f"teds={score.teds:.3f} teds_struct={score.teds_struct:.3f}"
        if score.error:
            status = f"error={score.error}"

        print(f"  {sample.sample_id[:48]}  {status}")

    if not scores:
        print("[torvex-bench] no scores produced — nothing to save.", file=sys.stderr)
        return 2

    summary = summarize_table_structure_scores(scores)

    # save per-sample scores
    scores_dir = score_output_dir / dataset / engine_label
    scores_dir.mkdir(parents=True, exist_ok=True)

    scores_jsonl = scores_dir / "fintabnet_scores.jsonl"
    save_table_structure_scores_jsonl(scores, scores_jsonl)

    # save run summary JSON
    summary_dict = {
        "dataset":            "docling-project/FinTabNet_OTSL",
        "dataset_slug":       "docling-project/FinTabNet_OTSL",
        "hf_dataset_commit":  get_hf_dataset_commit(),
        "engine":             engine,
        "engine_label":       engine_label,
        "samples_total":      summary.samples_total,
        "samples_error_free": summary.samples_error_free,
        "samples_missing":    missing,
        "mean_teds":          summary.mean_teds,
        "mean_teds_struct":   summary.mean_teds_struct,
        "missing_table_count": summary.missing_table_count,
        "error_count":        summary.error_count,
        "scored_at":          scored_at,
        "run_output_dir":     str(run_output_dir),
        "scores_jsonl":       str(scores_jsonl),
    }

    summary_path = scores_dir / "fintabnet_summary.json"
    summary_path.write_text(
        json.dumps(summary_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # print summary
    print()
    print(f"[torvex-bench] FinTabNet results — engine={engine_label}")
    print(f"  mean_teds        = {summary.mean_teds:.4f}")
    print(f"  mean_teds_struct = {summary.mean_teds_struct:.4f}")
    print(f"  samples_total    = {summary.samples_total}")
    print(f"  error_free       = {summary.samples_error_free}")
    print(f"  missing          = {missing}")
    print(f"  error_count      = {summary.error_count}")
    print()
    print(f"[torvex-bench] scores  → {scores_jsonl}")
    print(f"[torvex-bench] summary → {summary_path}")

    return 0 if (summary.error_count == 0 and missing == 0) else 1


DATASET_SCORERS: dict[str, Callable[..., int]] = {
    "fintabnet": score_fintabnet,
    # others added as scorers are implemented
}


# ── Parser ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="torvex-bench",
        description=(
            "Torvex Bench CLI. "
            "Two commands: run (extract PDFs) and score (evaluate outputs)."
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── run subcommand ────────────────────────────────────────────────────────
    run_parser = subparsers.add_parser(
        "run",
        help="Run one engine over one dataset. Extracts PDFs and saves outputs.",
    )

    run_parser.add_argument(
        "--dataset",
        choices=sorted(SUPPORTED_DATASETS),
        required=True,
        help="Dataset to prepare/load and run.",
    )

    run_parser.add_argument(
        "--engine",
        choices=sorted(SUPPORTED_ENGINES),
        default="torvex",
        help="Engine adapter to run. Only torvex is production-ready right now.",
    )

    run_parser.add_argument(
        "--limit",
        type=positive_int_or_none,
        default=None,
        help="Max samples to run. For olmOCR this means PDF samples not unit-test rows.",
    )

    run_parser.add_argument(
        "--split",
        default="test",
        help="Dataset split for FinTabNet/DocLayNet. Ignored by OmniDocBench/olmOCR.",
    )

    run_parser.add_argument(
        "--input-type",
        choices=["digital", "scanned"],
        default="digital",
        help=(
            "PDF mode for OmniDocBench. "
            "digital=ori_pdfs (text layer), scanned=pdfs (image-converted)."
        ),
    )

    run_parser.add_argument(
        "--olmocr-subset",
        choices=["non_math", "math", "all"],
        default="non_math",
        help="olmOCR subset. Core track=non_math. Formula track=math or all.",
    )

    run_parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=None,
        help="Override dataset raw/cache dir. Uses loader env var/default if omitted.",
    )

    run_parser.add_argument(
        "--dataset-output-dir",
        type=Path,
        default=None,
        help="Override dataset output dir. Uses loader env var/default if omitted.",
    )

    run_parser.add_argument(
        "--run-output-dir",
        type=Path,
        default=DEFAULT_RUN_OUTPUT_DIR,
        help="Root dir for runner raw/normalized/record JSON outputs.",
    )

    run_parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device knob recorded in run metadata. Affects engine warmup when consumed.",
    )

    run_parser.add_argument(
        "--enable-formula",
        action="store_true",
        help="Formula extraction knob recorded in run metadata.",
    )

    run_parser.add_argument(
        "--track",
        choices=["core", "formula"],
        default="core",
        help="core=practical finance extraction. formula=diagnostic math capability.",
    )

    run_parser.add_argument(
        "--no-download-pdfs",
        action="store_true",
        help="olmOCR only: skip PDF download. For unit tests only.",
    )

    run_parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing outputs. Use --no-overwrite to skip already-run samples.",
    )

    run_parser.add_argument(
        "--raise-on-error",
        action="store_true",
        help="Re-raise adapter errors. Useful for debugging.",
    )

    run_parser.set_defaults(func=run_command)

    # ── score subcommand ──────────────────────────────────────────────────────
    score_parser = subparsers.add_parser(
        "score",
        help=(
            "Score saved engine outputs against dataset ground truth. "
            "Reads normalized JSONs written by 'run'. Does not re-extract."
        ),
    )

    score_parser.add_argument(
        "--dataset",
        choices=sorted(SUPPORTED_DATASETS),
        required=True,
        help="Dataset to score.",
    )

    score_parser.add_argument(
        "--engine",
        choices=sorted(SUPPORTED_ENGINES),
        default="torvex",
        help="Engine whose outputs to score.",
    )

    score_parser.add_argument(
        "--limit",
        type=positive_int_or_none,
        default=None,
        help="Max samples to score. Must match or be less than the run limit.",
    )

    score_parser.add_argument(
        "--split",
        default="test",
        help="Dataset split. Must match what was used in 'run'.",
    )

    score_parser.add_argument(
        "--run-output-dir",
        type=Path,
        default=DEFAULT_RUN_OUTPUT_DIR,
        help="Root dir where runner saved normalized JSONs.",
    )

    score_parser.add_argument(
        "--score-output-dir",
        type=Path,
        default=DEFAULT_SCORE_OUTPUT_DIR,
        help="Root dir to save score JSONL + summary JSON.",
    )

    score_parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=None,
        help="Override dataset raw/cache dir for manifest loading.",
    )

    score_parser.add_argument(
        "--dataset-output-dir",
        type=Path,
        default=None,
        help="Override dataset output dir for manifest loading.",
    )

    score_parser.add_argument(
        "--olmocr-subset",
        choices=["non_math", "math", "all"],
        default="non_math",
        help="olmOCR subset. Must match what was used in 'run'.",
    )

    score_parser.add_argument(
        "--no-download-pdfs",
        action="store_true",
        help="olmOCR only: skip PDF download when loading manifest.",
    )

    score_parser.set_defaults(func=score_command)

    return parser


# ── Command handlers ──────────────────────────────────────────────────────────

def validate_run_args(args: argparse.Namespace) -> None:
    if args.no_download_pdfs and args.dataset != "olmocr":
        raise SystemExit("ERROR: --no-download-pdfs is only valid with --dataset olmocr")

    if args.enable_formula and args.track == "core":
        print(
            "WARNING: --enable-formula on --track core. "
            "Formula content not scored in core track. Recorded in metadata only.",
            file=sys.stderr,
        )

    if args.track == "formula" and args.dataset not in {"olmocr"}:
        print(
            f"WARNING: --track formula with --dataset {args.dataset}. "
            "Formula track is meaningful for olmOCR math/all only.",
            file=sys.stderr,
        )

    if args.dataset == "olmocr" and args.track == "formula" and args.olmocr_subset == "non_math":
        print(
            "WARNING: --track formula with --olmocr-subset non_math. "
            "Use --olmocr-subset math or all for formula track.",
            file=sys.stderr,
        )


def run_command(args: argparse.Namespace) -> int:
    validate_run_args(args)

    loader = DATASET_LOADERS[args.dataset]

    print(
        f"[torvex-bench] preparing dataset={args.dataset} "
        f"limit={args.limit if args.limit is not None else 'full'} ..."
    )

    samples = loader(args)
    print(f"[torvex-bench] loaded {len(samples)} samples")

    if not samples:
        print("[torvex-bench] no samples loaded — nothing to run.", file=sys.stderr)
        return 2

    adapter = make_adapter(args.engine)
    engine_label = getattr(adapter, "name", adapter.__class__.__name__)

    run_metadata = {
        "track":          args.track,
        "device":         args.device,
        "enable_formula": bool(args.enable_formula),
        "dataset":        args.dataset,
        "engine":         args.engine,
        "split":          args.split,
        "input_type":     args.input_type,
        "olmocr_subset":  args.olmocr_subset,
        "limit":          args.limit,
    }

    print(
        f"[torvex-bench] running "
        f"engine={engine_label} dataset={args.dataset} "
        f"input_type={args.input_type} device={args.device} track={args.track}"
    )

    records = run_samples(
        adapter=adapter,
        samples=samples,
        dataset=args.dataset,
        output_dir=args.run_output_dir,
        input_type=args.input_type,
        limit=args.limit,
        run_metadata=run_metadata,
        overwrite=args.overwrite,
        raise_on_error=args.raise_on_error,
    )

    summary = count_successes(records)

    print(
        f"[torvex-bench] done: "
        f"ok={summary['ok']} error={summary['error']} total={summary['total']}"
    )

    output_jsonl = args.run_output_dir / args.dataset / engine_label / "run_records.jsonl"
    print(f"[torvex-bench] run records → {output_jsonl}")

    return 0 if summary["error"] == 0 else 1


def score_command(args: argparse.Namespace) -> int:
    """
    Score saved engine outputs against dataset ground truth.

    Flow:
        1. Load dataset manifest → get samples with GT
        2. For each sample → find normalized JSON from run output
        3. Run dataset-specific scorer
        4. Save scores JSONL + summary JSON
        5. Print numbers

    Does not re-extract. Reads what runner already saved.
    """
    dataset = args.dataset
    engine = args.engine
    engine_label = _engine_label_from_name(engine)

    scored_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # check scorer exists for this dataset
    if dataset not in DATASET_SCORERS:
        print(
            f"[torvex-bench] scorer for dataset={dataset} is not implemented yet. "
            f"Implemented: {sorted(DATASET_SCORERS.keys())}",
            file=sys.stderr,
        )
        return 2

    # load samples (manifest only — no extraction)
    loader = DATASET_LOADERS[dataset]

    print(
        f"[torvex-bench] loading manifest dataset={dataset} "
        f"limit={args.limit if args.limit is not None else 'full'} ..."
    )

    samples = loader(args)
    print(f"[torvex-bench] loaded {len(samples)} samples")

    if not samples:
        print("[torvex-bench] no samples loaded — nothing to score.", file=sys.stderr)
        return 2

    scorer = DATASET_SCORERS[dataset]

    print(
        f"[torvex-bench] scoring "
        f"engine={engine_label} dataset={dataset} "
        f"run_output_dir={args.run_output_dir}"
    )

    return scorer(
        samples=samples,
        engine_label=engine_label,
        run_output_dir=args.run_output_dir,
        score_output_dir=args.score_output_dir,
        dataset=dataset,
        engine=engine,
        scored_at=scored_at,
    )


# ── Entry points ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def app() -> None:
    """
    Console-script entry point from pyproject.toml:

        torvex-bench = "torvex_bench.cli:app"
    """
    raise SystemExit(main())


if __name__ == "__main__":
    app()