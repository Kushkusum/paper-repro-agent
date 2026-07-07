from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_paper_text(paper_path: Path) -> str:
    if paper_path.suffix.lower() == ".pdf":
        return extract_text(paper_path)
    return paper_path.read_text(encoding="utf-8")
