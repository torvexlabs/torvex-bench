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