# Torvex Bench

**Reproducible benchmark harness for PDF extraction pipelines.**

`torvex-bench` compares **Torvex Extract** against **Docling** and **PPStructureV3** on document extraction tasks that matter for finance and audit workflows:

* text extraction
* table detection
* table structure extraction
* layout zone detection
* reading order
* OCR behavior
* formula-zone detection
* finance fact extraction using EDGAR/XBRL
* latency, RAM, and VRAM

This repository is part of **Torvex Labs**.

The commercial product platform, **ClearVault**, is separate and private. This repo only benchmarks extraction systems.

---

## Status

**Current status:** pre-build benchmark harness.

No final benchmark numbers are published yet.

This repository currently focuses on:

1. building adapters for each extraction pipeline
2. normalizing outputs into one schema
3. running dataset-specific scorers
4. producing reproducible score files, charts, and benchmark reports

Final numbers will be published only after all adapters, scorers, and dataset loaders are validated.

---

## Systems Compared

| System             | Role                                                 |
| ------------------ | ---------------------------------------------------- |
| **Torvex Extract** | Torvex Labs extraction engine                        |
| **Docling**        | IBM Research document conversion / extraction system |
| **PPStructureV3**  | PaddleOCR / Baidu document parsing pipeline          |

All systems are evaluated on the same inputs, same datasets, same scoring logic, and same hardware.

---

## Scope

### In scope

Torvex Bench evaluates:

* text extraction accuracy
* table extraction accuracy
* table structure accuracy
* layout detection accuracy
* reading order correctness
* OCR behavior on scanned / image-heavy documents
* figure, chart, image, and formula-zone detection
* EDGAR financial fact extraction using XBRL ground truth
* CPU and GPU latency
* peak RAM usage
* peak VRAM usage

### Out of scope

The benchmark does **not** evaluate:

* formula content extraction
* LaTeX reconstruction
* chart content extraction
* handwriting recognition
* spreadsheet-native extraction
* DOCX/PPTX/XLSX conversion

Formula **zones** are evaluated as bounding boxes.

Formula **content** is excluded from all systems.

---

## Formula Handling

Formula content metrics are excluded equally for all three systems.

The OmniDocBench composite score is:

```text
(text NED + table TEDS + reading order NED) / 3
```

Formula CDM / LaTeX scoring is not included.

This is intentional because Torvex Extract detects formula regions but does not attempt formula-to-LaTeX conversion.

Formula detection is still evaluated separately using bounding-box mAP where ground truth exists.

---

## Datasets

Torvex Bench uses public benchmark datasets plus one EDGAR/XBRL finance benchmark.

| Dataset               | Purpose                                                     |
| --------------------- | ----------------------------------------------------------- |
| **DocLayNet v1.2**    | layout, table-zone, formula-zone detection                  |
| **FinTabNet OTSL**    | finance table structure extraction                          |
| **OmniDocBench v1.6** | end-to-end text, table, and reading-order benchmark         |
| **olmOCR-bench**      | OCR / extraction unit-test pass rate                        |
| **EDGAR/XBRL corpus** | finance fact extraction against SEC structured ground truth |

Only test splits or fixed benchmark subsets are used.

Large training splits are not used.

---

## Dataset Notes

### DocLayNet v1.2

Used for layout detection.

DocLayNet contains document pages with annotated layout regions. It is used to score:

* overall layout mAP
* table-zone mAP
* figure / image-zone mAP
* formula-zone mAP
* finance document category mAP

### FinTabNet OTSL

Used for finance table structure extraction.

FinTabNet OTSL provides table-crop images and table ground truth. It is not a full-document extraction benchmark.

This benchmark uses it only for table structure scoring.

### OmniDocBench v1.6

Used as the main end-to-end extraction benchmark.

It scores:

* text NED
* table TEDS
* reading-order NED

Formula content scoring is excluded.

### olmOCR-bench

Used for OCR / extraction behavior.

olmOCR-bench uses unit-test-style checks instead of edit-distance metrics.

Primary score:

```text
unit test pass rate
```

CER and WER are not used for olmOCR.

### EDGAR/XBRL

Used for finance-domain fact extraction.

The source document is extracted by each pipeline. Ground truth comes from SEC XBRL data.

EDGAR metrics include:

* XBRL fact alignment
* numeric value recall
* label match rate
* numeric NED
* numeric float parity

EDGAR does not use TEDS, because XBRL provides semantic financial facts, not table-layout ground truth.

---

## Banned Dataset

### PubTables-1M

PubTables-1M is banned from this benchmark.

Reason:

Torvex Extract uses TATR for table structure extraction, and TATR was trained on PubTables-1M. Benchmarking on PubTables-1M would create training-data leakage.

Scores from that dataset would be misleading and are not used.

---

## Repository Structure

```text
torvex-bench/
│
├── src/torvex_bench/
│   ├── adapters/
│   │   ├── base.py
│   │   ├── torvex_extract_adapter.py
│   │   ├── docling_adapter.py
│   │   └── ppstructure_adapter.py
│   │
│   ├── datasets/
│   │   ├── base.py
│   │   ├── doclaynet.py
│   │   ├── fintabnet.py
│   │   ├── omnidocbench.py
│   │   ├── olmocr_bench.py
│   │   └── edgar.py
│   │
│   ├── scorers/
│   │   ├── layout.py
│   │   ├── table_detection.py
│   │   ├── table_structure.py
│   │   ├── text.py
│   │   ├── reading_order.py
│   │   ├── ocr.py
│   │   ├── edgar_xbrl.py
│   │   └── profiler.py
│   │
│   ├── converters/
│   │   └── edgar_html_to_pdf.py
│   │
│   ├── normalizer.py
│   ├── runner.py
│   ├── report.py
│   └── cli.py
│
├── results/
│   ├── raw/
│   ├── scores/
│   └── summaries/
│
├── charts/
├── sample_outputs/
├── benchmark.md
├── methodology.md
├── pyproject.toml
└── README.md
```

---

## Result Storage

Torvex Bench separates raw machine outputs from human-readable benchmark proof.

```text
results/raw/          compressed raw predictions, not committed
results/scores/       small score JSON files, committed
results/summaries/    markdown and CSV summaries, committed
charts/               generated benchmark charts, committed
benchmark.md          final published leaderboard
methodology.md        full evaluation methodology
```

Raw prediction files may be large, so they are stored as `.jsonl.gz` and excluded from git.

Only scores, summaries, charts, and methodology are committed.

---

## Why JSONL.gz?

Full extraction output can be large.

A single page may contain:

* final text
* tables
* layout zones
* bounding boxes
* timing metadata
* warnings
* OCR path metadata

For full benchmark runs, Torvex Bench writes compact normalized records to compressed JSONL:

```text
one page/result per line
streamable
resumable
compressed
safe for large datasets
```

This avoids giant in-memory JSON files.

---

## Normalized Output Contract

Every adapter converts its pipeline output into the same internal result format.

The benchmark does not score raw Torvex, Docling, or PPStructure outputs directly.

Flow:

```text
pipeline raw output
        ↓
adapter
        ↓
DocumentResult / PageResult / TableResult
        ↓
normalizer
        ↓
scorers
        ↓
score JSON
        ↓
charts + benchmark.md
```

This keeps scoring fair and pipeline-agnostic.

---

## Adapter Contract

Each pipeline implements one document-level adapter.

Adapters must extract a full document once.

Per-page extraction is not allowed because it would repeatedly process the same PDF and distort latency numbers.

```python
class ExtractionAdapter:
    def extract_document(self, pdf_path: str) -> DocumentResult:
        """
        Extract a full document once.
        Return all page-level results.
        """
        raise NotImplementedError
```

The normalizer handles page-level splitting downstream.

---

## Torvex Extract Adapter

The Torvex Extract adapter calls the real Phase 1 extraction entry point:

```python
extract_with_pypdfium2(pdf_path)
```

It maps Phase 1 fields into benchmark fields:

| Phase 1 field          | Benchmark field             |
| ---------------------- | --------------------------- |
| `page["final_text"]`   | `PageResult.text`           |
| `page["tables"]`       | `PageResult.tables`         |
| `table["rows"]`        | `TableResult.rows`          |
| `table["bbox_pdfium"]` | `TableResult.bbox_pdfium`   |
| `page["zones"]`        | `PageResult.layout_zones`   |
| formula zone types     | `PageResult.formula_bboxes` |
| `page["needs_ocr"]`    | `PageResult.needs_ocr`      |

If `page["formula_bboxes"]` does not exist, formula bboxes are derived from layout zones:

```text
display_formula
inline_formula
formula_number
```

---

## Metrics

### Headline Metrics

| Task                       | Metric              |
| -------------------------- | ------------------- |
| Layout detection           | mAP@0.5             |
| Formula-zone detection     | mAP@0.5             |
| Table detection            | IoU@0.5             |
| Table structure            | TEDS / TEDS-Struct  |
| OmniDocBench text          | NED                 |
| OmniDocBench reading order | NED                 |
| olmOCR                     | unit test pass rate |
| EDGAR finance facts        | XBRL fact alignment |
| Latency                    | p50 / p95 / p99     |
| RAM                        | peak RSS            |
| VRAM                       | peak VRAM           |

### Secondary Diagnostics

Secondary metrics may be reported but are not headline scores:

* CER
* WER
* Kendall Tau
* failure rate
* skipped sample rate
* float-parity skipped rate

---

## Reproducibility Metadata

Every score file includes metadata.

Required fields:

```json
{
  "meta": {
    "torvex_extract_commit": "...",
    "benchmark_commit": "...",
    "python_version": "...",
    "os": "...",
    "cpu": "...",
    "gpu": "...",
    "ram_gb": 32,
    "batch_size": 1,
    "render_dpi": 200,
    "formula_cdm_excluded": true,
    "composite_formula": "(text_ned + table_teds + reading_order_ned) / 3",
    "model_versions": {
      "pp_doclayout": "PP-DocLayoutV3_ir8.onnx",
      "tatr": "tatr-v1.1-all.onnx",
      "onnxtr": "..."
    },
    "model_sha256": {
      "pp_doclayout_onnx": "sha256:...",
      "tatr_onnx": "sha256:..."
    },
    "run_date": "..."
  }
}
```

Model SHA256 hashes are required because model files can change silently between downloads.

---

## Installation

During development, install from source:

```bash
git clone https://github.com/torvexlabs/torvex-bench.git
cd torvex-bench
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Windows PowerShell:

```powershell
git clone https://github.com/torvexlabs/torvex-bench.git
cd torvex-bench
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Optional dependencies:

```bash
pip install -e ".[eval]"
pip install -e ".[docling]"
pip install -e ".[ppstructure]"
```

Install everything:

```bash
pip install -e ".[all]"
```

---

## Optional Dependency Groups

| Extra         | Purpose                                        |
| ------------- | ---------------------------------------------- |
| `eval`        | docling-eval and shared evaluation tooling     |
| `docling`     | run Docling comparison                         |
| `ppstructure` | run PPStructureV3 comparison                   |
| `all`         | install all comparison/evaluation dependencies |

PPStructureV3 may require separate PaddlePaddle setup depending on CPU/GPU environment.

---

## Development Flow

### Step 0 — Inspect datasets

Before writing adapters or scorers, inspect dataset schemas.

```bash
python scripts/count_dataset_splits.py
```

The output becomes part of `methodology.md`.

Dataset field names are verified at build time instead of guessed.

### Step 1 — Validate sample outputs

Confirm the real Torvex Extract / Phase 1 output schema using:

```text
sample_outputs/apple_sample_output.json
sample_outputs/invoice_sample_output.json
```

These files are used to build the Torvex Extract adapter without guessing field names.

### Step 2 — Build adapters

Build:

```text
torvex_extract_adapter.py
docling_adapter.py
ppstructure_adapter.py
```

All adapters must output the same `DocumentResult` schema.

### Step 3 — Build fastest feedback benchmark

Start with FinTabNet.

Reason:

* smaller scope
* table-focused
* finance-relevant
* fastest way to validate adapter → normalizer → scorer

Run first 100 tables before full benchmark.

### Step 4 — Add remaining datasets

Then add:

```text
DocLayNet
OmniDocBench
EDGAR
olmOCR
```

### Step 5 — Run local dev samples

Use small sample limits:

```bash
torvex-bench run --dataset fintabnet --system torvex --limit 100
torvex-bench run --dataset doclaynet --system torvex --limit 50
```

### Step 6 — Final runs

Final benchmark numbers must come from one consistent hardware setup.

Planned final setup:

```text
RunPod RTX 4090
batch_size = 1
render_dpi = 200
same hardware for all systems
same formula exclusion for all systems
```

---

## Build Order

```text
Step 0   Run dataset split inspection
Step 1   Confirm sample outputs exist
Step 2   Write adapter base and Torvex Extract adapter
Step 3   Write Docling and PPStructureV3 adapters
Step 4   Write FinTabNet loader and TEDS scorer
Step 5   Run 100 FinTabNet samples
Step 6   Write DocLayNet loader and mAP scorer
Step 7   Write OmniDocBench loader with formula content excluded
Step 8   Write EDGAR loader, HTML renderer, and XBRL scorer
Step 9   Write olmOCR unit-test scorer
Step 10  Write profiler
Step 11  Write normalizer, report generator, and charts
Step 12  Run local dev samples
Step 13  Run incremental full datasets
Step 14  Run final benchmark on RunPod
Step 15  Commit scores, summaries, charts, benchmark.md
Step 16  Tag GitHub release
```

---

## Failure Handling

A benchmark run should not stop because one page fails.

Failed pages are written to:

```text
results/raw/failed/
```

Every score file reports:

```text
failure_rate
failed_pages
processed_pages
skipped_pages
```

A pipeline that silently drops data is worse than a pipeline that reports failures.

Failure rate is a published metric.

---

## EDGAR/XBRL Benchmark

EDGAR is the finance-domain differentiator.

The benchmark flow:

```text
SEC filing
  ↓
native PDF or rendered HTML/XHTML PDF
  ↓
pipeline extraction
  ↓
SEC XBRL facts as ground truth
  ↓
fact alignment + numeric scoring
```

EDGAR uses all three systems:

```text
Torvex Extract
Docling
PPStructureV3
```

This makes it a fair finance-domain comparison.

### EDGAR scoring

EDGAR scores:

* whether a fact was extracted
* whether the correct numeric value was extracted
* whether the value was linked to the correct label
* whether numeric text matches after normalization
* whether parsed floats match within tolerance

TEDS is not used for EDGAR because XBRL is semantic ground truth, not layout ground truth.

---

## Hardware Policy

All final numbers must be generated on one consistent machine.

Do not mix:

```text
local RTX 3070 numbers
Kaggle T4 numbers
RunPod RTX 4090 numbers
```

in the same final table.

Kaggle is used for incremental reproducibility checks.

RunPod RTX 4090 is used for final published numbers.

---

## Charts

Generated charts include:

* FinTabNet TEDS comparison
* DocLayNet mAP comparison
* formula-zone mAP comparison
* OmniDocBench text NED comparison
* latency comparison
* RAM comparison
* VRAM comparison
* accuracy vs latency scatter plot

Charts are generated from `results/scores/*.json`, not raw prediction files.

---

## Published Benchmark Table

The final benchmark table will be written to:

```text
benchmark.md
```

It will include:

* dataset name
* metric name
* Torvex Extract score
* Docling score
* PPStructureV3 score
* hardware
* run date
* reproduction instructions
* formula exclusion note

No benchmark score is published without a matching methodology entry.

---

## Methodology

`methodology.md` documents:

* dataset versions
* split sizes
* input format handling
* metrics
* formula exclusion
* hardware
* software versions
* model hashes
* failure policy
* scoring pipeline
* EDGAR/XBRL construction
* limitations

This file is part of the benchmark proof.

---

## Limitations

Current known limitations:

* formula content extraction is not evaluated
* chart content extraction is not evaluated
* FinTabNet is table-crop only, not full-page document extraction
* EDGAR XBRL scoring measures financial facts, not table layout
* Office formats are out of scope
* XLSX native extraction will be handled separately from PDF extraction
* ClearVault product code is not part of this benchmark repo

---

## Research Plan

Torvex Bench is intended to support an arXiv technical report.

Working title:

```text
Torvex Extract: A Finance-Domain PDF Extraction Pipeline with Formula-Zone Detection Benchmarked Against Docling and PPStructureV3
```

Primary contributions:

1. finance-domain PDF extraction benchmark with EDGAR/XBRL ground truth
2. head-to-head evaluation against Docling and PPStructureV3
3. formula-zone detection without formula content extraction
4. reproducible benchmark code and score artifacts

---

## Relationship to ClearVault

ClearVault is the private product platform.

Torvex Bench does not include:

* ClearVault UI
* ClearVault database
* ClearVault retrieval pipeline
* customer deployment code
* enterprise product logic

Torvex Bench only evaluates extraction engines.

---

## License

License is not finalized yet.

Before public release, choose one:

```text
Apache-2.0
MIT
```

For benchmark/research tooling, Apache-2.0 is recommended.

---

## Citation

Citation will be added after the arXiv technical report is published.

Placeholder:

```bibtex
@misc{torvexextract2026,
  title={Torvex Extract: A Finance-Domain PDF Extraction Pipeline with Formula-Zone Detection Benchmarked Against Docling and PPStructureV3},
  author={Torvex Labs},
  year={2026},
  note={arXiv preprint forthcoming}
}
```

---

## Current Build Stage

```text
Current step:
Step 0 — inspect dataset schemas and finalize count_dataset_splits.py
```

No final scores have been published yet.
