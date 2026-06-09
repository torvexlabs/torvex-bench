from __future__ import annotations

import pytest

from torvex_bench.cli import _resolve_formula_override, build_parser


def test_omnidocbench_formula_flags_parse() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "official-omnidocbench",
            "--limit",
            "1",
            "--enable-formula",
        ]
    )

    assert args.enable_formula is True
    assert args.disable_formula is False


def test_olmocr_formula_flags_parse() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "official-olmocr",
            "--limit",
            "1",
            "--track",
            "math",
            "--disable-formula",
        ]
    )

    assert args.enable_formula is False
    assert args.disable_formula is True


def test_formula_override_rejects_conflicting_flags() -> None:
    with pytest.raises(ValueError, match="Use only one"):
        _resolve_formula_override(
            enable_formula=True,
            disable_formula=True,
        )


def test_formula_override_default_is_none() -> None:
    assert _resolve_formula_override(
        enable_formula=False,
        disable_formula=False,
    ) is None