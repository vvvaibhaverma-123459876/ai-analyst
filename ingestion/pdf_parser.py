"""
ingestion/pdf_parser.py
Handles: .pdf
Extracts: text (via pdfminer/pypdf), tables (via pdfplumber), metadata.
Falls back gracefully if optional deps not installed.
"""

from __future__ import annotations
import io
import pandas as pd
from ingestion.base_parser import BaseParser, ParsedDocument
from core.logger import get_logger

logger = get_logger(__name__)


class PDFParser(BaseParser):

    def can_parse(self, filename: str, mime_type: str = "") -> bool:
        return filename.lower().endswith(".pdf") or "pdf" in mime_type

    def parse(self, source, filename: str = "") -> ParsedDocument:
        doc = ParsedDocument(source_name=filename, source_type="pdf")

        # Normalise source to bytes
        if hasattr(source, "read"):
            pdf_bytes = source.read()
        elif isinstance(source, (str, bytes)):
            if isinstance(source, str):
                with open(source, "rb") as f:
                    pdf_bytes = f.read()
            else:
                pdf_bytes = source
        else:
            doc.add_warning("PDF source type not recognised.")
            return doc

        # ── Text extraction ──────────────────────────────────────────
        text = self._extract_text(pdf_bytes, doc)
        if text:
            # Split into logical chunks (~500 words each)
            words = text.split()
            chunk_size = 500
            for i in range(0, len(words), chunk_size):
                doc.text_chunks.append(" ".join(words[i:i + chunk_size]))
            logger.info(f"PDF text: {len(words)} words → {len(doc.text_chunks)} chunks")

        # ── Table extraction ─────────────────────────────────────────
        self._extract_tables(pdf_bytes, doc)

        # ── Metadata ─────────────────────────────────────────────────
        doc.metadata["filename"] = filename
        doc.metadata["text_length"] = len(text)
        doc.metadata["tables_found"] = len(doc.dataframes)

        return doc

    def _extract_text(self, pdf_bytes: bytes, doc: ParsedDocument) -> str:
        # Try pdfplumber first (best quality)
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                pages = []
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        pages.append(t)
                return "\n\n".join(pages)
        except ImportError:
            pass
        except Exception as e:
            doc.add_warning(f"pdfplumber text extraction failed: {e}")

        # Fallback: pypdf
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            pages = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    pages.append(t)
            return "\n\n".join(pages)
        except ImportError:
            doc.add_warning("PDF text extraction requires pdfplumber or pypdf. "
                            "Run: pip install pdfplumber")
        except Exception as e:
            doc.add_warning(f"pypdf text extraction failed: {e}")

        return ""

    def _extract_tables(self, pdf_bytes: bytes, doc: ParsedDocument):
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    for t_idx, table in enumerate(tables):
                        if not table or len(table) < 2:
                            continue
                        try:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            df = df.dropna(how="all").reset_index(drop=True)
                            if not df.empty:
                                doc.dataframes.append(df)
                                doc.table_names.append(f"page{page_num+1}_table{t_idx+1}")
                        except Exception as e:
                            doc.add_warning(f"Table extraction failed p{page_num}: {e}")
        except ImportError:
            pass
        except Exception as e:
            doc.add_warning(f"pdfplumber table extraction failed: {e}")
