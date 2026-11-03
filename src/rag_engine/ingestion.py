"""Load supported local documents into normalized text records."""

from pathlib import Path
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_document(path: str | Path) -> list[dict]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    records = []
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                records.append({
                    "text": text,
                    "metadata": {"source": str(path), "page": page_number},
                })
    else:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            records.append({"text": text, "metadata": {"source": str(path)}})
    return records


def load_directory(directory: str | Path) -> list[dict]:
    directory = Path(directory)
    if not directory.exists() or not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    documents = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.extend(load_document(path))
    return documents
