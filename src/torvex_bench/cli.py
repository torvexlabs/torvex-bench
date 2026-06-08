from __future__ import annotations

import argparse
from pathlib import Path


def positive_int(value: str) -> int:
    """
    Parse a positive integer CLI argument.

    Used for --limit so users cannot accidentally run with 0 or negative samples.
    """
    parsed = int(value)

    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")

    return parsed


def official_fintabnet_command(args: argparse.Namespace) -> int:
    """
    Run the official FinTabNet benchmark flow.

    CLI responsibility:
        - read parsed args
        - call production benchmark runner
        - print compact result
        - return process exit code

    Actual benchmark logic lives in:
        torvex_bench.harnesses.official_fintabnet
    """
    from torvex_bench.harnesses.official_fintabnet import (
        DEFAULT_WORK_DIR,
        run_official_fintabnet,
    )

    work_dir = args.work_dir or DEFAULT_WORK_DIR

    summary = run_official_fintabnet(
        limit=args.limit,
        work_dir=work_dir,
        clean=args.clean,
        save_normalized=args.save_normalized,
    )

    print()
    print("[torvex-bench] official FinTabNet result")
    print(f"  limit             = {summary.limit}")
    print(f"  evaluated_samples = {summary.evaluated_samples}")
    print(f"  TEDS mean         = {summary.teds_mean:.4f}")
    print(f"  TEDS_struct mean  = {summary.teds_struct_mean:.4f}")
    print(f"  rejected          = {summary.rejected_samples}")
    print(f"  official_result   = {summary.official_result_path}")
    print(f"  summary           = {summary.summary_path}")

    rejected_total = sum(summary.rejected_samples.values())
    prediction_errors = summary.prediction_summary.errors

    return 0 if rejected_total == 0 and prediction_errors == 0 else 1


def official_doclaynet_command(args: argparse.Namespace) -> int:
    """
    Run the official DocLayNetV1 benchmark flow.

    CLI responsibility:
        - read parsed args
        - call production benchmark runner
        - print compact result
        - return process exit code

    Actual benchmark logic lives in:
        torvex_bench.harnesses.official_doclaynet
    """
    from torvex_bench.harnesses.official_doclaynet import (
        DEFAULT_WORK_DIR,
        run_official_doclaynet,
    )

    work_dir = args.work_dir or DEFAULT_WORK_DIR

    summary = run_official_doclaynet(
        limit=args.limit,
        work_dir=work_dir,
        clean=args.clean,
        save_normalized=args.save_normalized,
        device=args.device,
    )

    print()
    print("[torvex-bench] official DocLayNetV1 result")
    print(f"  limit                = {summary.limit}")
    print(f"  device               = {summary.device}")
    print(f"  evaluated_samples    = {summary.evaluated_samples}")
    print(f"  mAP                  = {summary.map_score:.4f}")
    print(f"  map_50 mean          = {summary.map_50_mean:.4f}")
    print(f"  map_75 mean          = {summary.map_75_mean:.4f}")
    print(f"  weighted_map_50 mean = {summary.weighted_map_50_mean:.4f}")
    print(f"  rejected             = {summary.rejected_samples}")
    print(f"  prediction_errors    = {summary.prediction_summary.errors}")
    print(f"  official_result      = {summary.official_result_path}")
    print(f"  summary              = {summary.summary_path}")

    rejected_total = sum(summary.rejected_samples.values())
    prediction_errors = summary.prediction_summary.errors

    return 0 if rejected_total == 0 and prediction_errors == 0 else 1


def official_omnidocbench_command(args: argparse.Namespace) -> int:
    """
    Run the official OmniDocBench scanned/image-page benchmark flow.

    CLI responsibility:
        - read parsed args
        - call production official OmniDocBench runner
        - print compact result
        - return process exit code

    Actual benchmark logic lives in:
        torvex_bench.harnesses.official_omnidocbench

    Locked scope:
        - scanned/image-page path only
        - text_block Edit_dist
        - table TEDS
        - reading_order Edit_dist
        - formula CDM omitted
        - COCO Det mAP omitted from this end-to-end path
    """
    from torvex_bench.harnesses.official_omnidocbench import (
        DEFAULT_WORK_DIR,
        run_official_omnidocbench,
    )

    work_dir = args.work_dir or DEFAULT_WORK_DIR

    summary = run_official_omnidocbench(
        limit=args.limit,
        work_dir=work_dir,
        clean=args.clean,
        save_normalized=args.save_normalized,
        device=args.device,
        eval_bin=args.eval_bin,
    )

    prediction_summary = summary.prediction_summary

    print()
    print("[torvex-bench] official OmniDocBench scanned result")
    print(f"  limit                     = {summary.limit}")
    print(f"  device                    = {args.device}")
    print(f"  predictions_written       = {prediction_summary.get('predictions_written')}")
    print(f"  empty_predictions_written = {prediction_summary.get('empty_predictions_written')}")
    print(f"  skipped_existing          = {prediction_summary.get('skipped_existing')}")
    print(f"  prediction_errors         = {prediction_summary.get('errors')}")
    print(f"  official_result           = {summary.official_result_path}")
    print(f"  official_run_summary      = {summary.official_run_summary_path}")
    print(f"  summary                   = {summary.summary_path}")

    prediction_errors = int(prediction_summary.get("errors") or 0)

    return 0 if prediction_errors == 0 else 1


def official_olmocr_command(args: argparse.Namespace) -> int:
    """
    Run the official olmOCR-Bench benchmark flow.

    CLI responsibility:
        - read parsed args
        - call production official olmOCR runner
        - print compact result
        - return process exit code

    Actual benchmark logic lives in:
        torvex_bench.harnesses.official_olmocr

    Scope:
        - default track is non_math
        - math/full tracks are explicit diagnostics
        - official evaluator computes unit-test pass rate
    """
    from torvex_bench.harnesses.official_olmocr import (
        DEFAULT_WORK_DIR,
        run_official_olmocr,
    )

    work_dir = args.work_dir or DEFAULT_WORK_DIR

    summary = run_official_olmocr(
        limit=args.limit,
        work_dir=work_dir,
        track=args.track,
        clean=args.clean,
        save_normalized=args.save_normalized,
        device=args.device,
        python_bin=args.python_bin,
    )

    prediction_summary = summary.prediction_summary

    print()
    print("[torvex-bench] official olmOCR-Bench result")
    print(f"  limit                     = {summary.limit}")
    print(f"  track                     = {summary.track}")
    print(f"  device                    = {summary.device}")
    print(f"  predictions_written       = {prediction_summary.get('predictions_written')}")
    print(f"  empty_predictions_written = {prediction_summary.get('empty_predictions_written')}")
    print(f"  skipped_existing          = {prediction_summary.get('skipped_existing')}")
    print(f"  prediction_errors         = {prediction_summary.get('errors')}")
    print(f"  eval_returncode           = {summary.eval_returncode}")
    print(f"  average_score             = {summary.average_score}")
    print(f"  total_tests               = {summary.total_tests}")
    print(f"  summary                   = {summary.summary_path}")
    print(f"  stdout                    = {summary.stdout_path}")
    print(f"  stderr                    = {summary.stderr_path}")

    prediction_errors = int(prediction_summary.get("errors") or 0)

    return 0 if prediction_errors == 0 and summary.eval_returncode == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    """
    Build the torvex-bench CLI parser.

    Design rule:
        cli.py should stay thin.
        It should only parse arguments and call production functions.
    """
    parser = argparse.ArgumentParser(
        prog="torvex-bench",
        description="Torvex Bench official benchmark CLI.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    fintabnet_parser = subparsers.add_parser(
        "official-fintabnet",
        help="Run official docling-eval FinTabNet table-structure benchmark.",
    )

    fintabnet_parser.add_argument(
        "--limit",
        type=positive_int,
        default=25,
        help="Number of FinTabNet test samples to run.",
    )

    fintabnet_parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help=(
            "Generated benchmark work directory. "
            "Default: benchmarks/docling_eval/FinTabNet_dev"
        ),
    )

    fintabnet_parser.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clean the work directory before running. Use --no-clean to keep it.",
    )

    fintabnet_parser.add_argument(
        "--save-normalized",
        action="store_true",
        help="Debug only: save normalized Torvex JSON under results/raw.",
    )

    fintabnet_parser.set_defaults(func=official_fintabnet_command)

    doclaynet_parser = subparsers.add_parser(
        "official-doclaynet",
        help="Run official docling-eval DocLayNetV1 layout benchmark.",
    )

    doclaynet_parser.add_argument(
        "--limit",
        type=positive_int,
        default=25,
        help="Number of DocLayNetV1 test samples to run.",
    )

    doclaynet_parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help=(
            "Generated benchmark work directory. "
            "Default: benchmarks/docling_eval/DocLayNetV1_dev"
        ),
    )

    doclaynet_parser.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clean the work directory before running. Use --no-clean to keep it.",
    )

    doclaynet_parser.add_argument(
        "--save-normalized",
        action="store_true",
        help="Debug only: save normalized Torvex JSON next to predictions.",
    )

    doclaynet_parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Torvex Extract ONNX inference device.",
    )

    doclaynet_parser.set_defaults(func=official_doclaynet_command)

    omnidocbench_parser = subparsers.add_parser(
        "official-omnidocbench",
        help="Run official OmniDocBench scanned/image-page end-to-end benchmark.",
    )

    omnidocbench_parser.add_argument(
        "--limit",
        type=positive_int,
        default=3,
        help="Number of OmniDocBench scanned/image samples to run.",
    )

    omnidocbench_parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help=(
            "Generated benchmark work directory. "
            "Default: benchmarks/omnidocbench/OmniDocBench_scanned"
        ),
    )

    omnidocbench_parser.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clean generated OmniDocBench outputs before running. Use --no-clean to keep them.",
    )

    omnidocbench_parser.add_argument(
        "--save-normalized",
        action="store_true",
        help="Debug only: save normalized Torvex JSON next to predictions.",
    )

    omnidocbench_parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Torvex Extract ONNX inference device.",
    )

    omnidocbench_parser.add_argument(
        "--eval-bin",
        type=Path,
        default=None,
        help=(
            "Path to omnidocbench-eval executable. "
            "Default: data/venvs/omnidocbench/Scripts/omnidocbench-eval.exe "
            "or OMNIDOCBENCH_EVAL_BIN."
        ),
    )

    omnidocbench_parser.set_defaults(func=official_omnidocbench_command)

    olmocr_parser = subparsers.add_parser(
        "official-olmocr",
        help="Run official olmOCR-Bench unit-test benchmark.",
    )

    olmocr_parser.add_argument(
        "--limit",
        type=positive_int,
        default=3,
        help="Number of olmOCR-Bench PDF samples to run.",
    )

    olmocr_parser.add_argument(
        "--track",
        choices=["non_math", "math", "full"],
        default="non_math",
        help=(
            "olmOCR-Bench track. "
            "Default non_math excludes arxiv_math and old_scans_math."
        ),
    )

    olmocr_parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help=(
            "Generated benchmark work directory. "
            "Default: benchmarks/olmocr/olmOCR_Bench_non_math"
        ),
    )

    olmocr_parser.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clean generated olmOCR outputs before running. Use --no-clean to keep them.",
    )

    olmocr_parser.add_argument(
        "--save-normalized",
        action="store_true",
        help="Debug only: save normalized Torvex JSON next to predictions.",
    )

    olmocr_parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Torvex Extract ONNX inference device.",
    )

    olmocr_parser.add_argument(
        "--python-bin",
        type=Path,
        default=None,
        help=(
            "Path to isolated olmOCR Python executable. "
            "Default: data/venvs/olmocr/Scripts/python.exe or OLMOCR_PYTHON."
        ),
    )

    olmocr_parser.set_defaults(func=official_olmocr_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Parse CLI args and dispatch to the selected command.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def app() -> None:
    """
    Console-script entry point from pyproject.toml.

    pyproject.toml should contain:
        torvex-bench = "torvex_bench.cli:app"
    """
    raise SystemExit(main())


if __name__ == "__main__":
    app()