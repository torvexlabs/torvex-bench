from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TableResult:
    rows: list[list[str]]
    bbox_pdfium: list[float] | None = None
    source: str = "unknown"
    confidence: float = 1.0
    bbox_plumber: list[float] | None = None
    bbox_px: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PageResult:
    page_num: int
    text: str = ""
    tables: list[TableResult] = field(default_factory=list)
    layout_zones: list[dict[str, Any]] = field(default_factory=list)
    formula_bboxes: list[list[float]] = field(default_factory=list)
    spotlight_bboxes: list[list[float]] = field(default_factory=list)
    needs_ocr: bool = False
    ocr_used: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentResult:
    pdf_path: str
    pages: list[PageResult] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ExtractionAdapter(ABC):
    @abstractmethod
    def extract_document(self, pdf_path: str) -> DocumentResult:
        """
        Extract one full PDF and return a benchmark-standard DocumentResult.

        Important:
            Always extract the full document once.
            Do not call the extractor page-by-page.
        """
        raise NotImplementedError

    # FIX: extract() alias lives on the ABC so every adapter exposes it.
    # Previously only TorvexExtractAdapter had this method — calling .extract()
    # on the Docling or PPStructure adapters would AttributeError at runtime
    # when runner.py loops over adapters generically.
    def extract(self, pdf_path: str) -> DocumentResult:
        """Convenience alias for extract_document(). Do not override."""
        return self.extract_document(pdf_path)