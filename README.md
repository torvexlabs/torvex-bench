# Torvex Bench

Reproducible benchmark harness for Torvex Extract.

`torvex-bench` is the evaluation layer for the Torvex document AI stack. It prepares public benchmark datasets, runs Torvex Extract, exports predictions in the format expected by official scorers, and stores compact summaries that can be checked later.

This repo is not a competitor leaderboard. That framing is wrong because the purpose is to prove the Torvex engine with reproducible benchmark evidence, not to publish marketing comparisons before the harnesses and artifacts are stable.

## Current Status

Active development. No final public benchmark scores should be treated as official unless they are backed by:

- the command used to generate the run
- the Torvex Extract commit
- the Torvex Bench commit
- the dataset split or fixed sample range
- the official evaluator output
- the compact summary written by this repo
- the hardware/runtime profile when latency or memory is discussed

If one of those fields is missing, the number is not publishable.

## What It Benchmarks

Torvex Bench currently focuses on Torvex Extract across public document AI benchmarks:

| Benchmark | Purpose | Scorer |
| --- | --- | --- |
| FinTabNet OTSL | table structure extraction | `docling-eval` official table evaluator |
| DocLayNetV1 | layout detection | `docling-eval` official layout evaluator |
| OmniDocBench | scanned/image-page end-to-end parsing | official OmniDocBench evaluator |
| olmOCR-Bench | OCR/document extraction behavior | official olmOCR-Bench evaluator |

`docling-eval` is used as an official evaluation harness and file format target where appropriate. External evaluator dependencies are tools here, not the main subject of the repo.

## What It Does Not Do

- It does not implement the extraction engine. That lives in `torvex-extract`.
- It does not contain ClearVault product code.
- It does not replace official benchmark scorers with custom metrics.
- It does not publish final scores without reproducible artifacts.
- It does not use PubTables-1M as a public benchmark for Torvex table structure, because TATR was trained on PubTables-1M and that would create leakage.

## Repository Boundary

The intended stack is:

```text
torvex-extract
  -> runs PDF extraction
  -> returns normalized document output

torvex-bench
  -> selects benchmark samples
  -> calls torvex-extract
  -> exports official prediction formats
  -> calls official evaluators
  -> writes summaries and profiling artifacts
```

Benchmark code should stay here. Engine code should stay in `torvex-extract`.

## Install

Clone `torvex-bench` beside `torvex-extract`:

```powershell
git clone https://github.com/torvexlabs/torvex-extract.git
git clone https://github.com/torvexlabs/torvex-bench.git
cd torvex-bench
uv sync
```

The local development setup expects `../torvex-extract` to exist. This is intentional because benchmarks should test the engine code you are actively developing.

Formula-enabled runs depend on the Torvex formula stack, including the UniMERNet ONNX runtime. On Windows/CUDA, verify that `CUDAExecutionProvider` is actually active before making GPU claims.

## CLI

```powershell
uv run torvex-bench --help
```

Available benchmark entry points:

```powershell
uv run torvex-bench official-fintabnet --limit 25
uv run torvex-bench official-doclaynet --limit 25
uv run torvex-bench official-omnidocbench --limit 3
uv run torvex-bench official-olmocr --limit 3
```

Useful options vary by benchmark:

- `--limit`: run a fixed number of samples for smoke or subset evaluation
- `--work-dir`: choose where generated benchmark artifacts are written
- `--clean` / `--no-clean`: control whether previous generated artifacts are removed
- `--device cpu|gpu`: select Torvex Extract inference device where supported
- `--ocr-backend onnxtr_fast_base|ppocrv6_small`: select OCR backend for OmniDocBench
- `--enable-formula` / `--disable-formula`: force formula extraction behavior where supported

## Output Policy

Generated benchmark data, predictions, temporary PDFs, and official evaluator outputs are local artifacts. They should not be casually committed.

Final publishable evidence should be small and reviewable:

- compact score summaries
- runtime summaries
- methodology notes
- selected charts
- exact command lines
- commit hashes

Large generated folders belong outside Git unless a specific artifact is intentionally published.

## Benchmark Rules

- Use official scorers wherever possible.
- Keep official scores separate from engine-only runtime profiling.
- Label smoke runs, subset runs, and full runs separately.
- Do not compare CPU and GPU numbers without naming hardware and providers.
- Do not claim "latest" results unless the run artifacts are current.
- Do not tune on a benchmark subset and then report that same subset as unbiased.

## Maintainer

Built and maintained by [Sibisrinivas B](https://github.com/sibisrinivasb) as part of [Torvex Labs](https://github.com/torvexlabs).

## License

Apache-2.0. See [LICENSE](LICENSE).
