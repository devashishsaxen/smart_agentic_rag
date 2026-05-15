import argparse
import re
from datetime import datetime
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    UPLOAD_DIR,
)


_embedding_model = None
PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_DIR.parent


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_paragraphs(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n")
    paragraphs = [normalize_text(part) for part in re.split(r"\n\s*\n", normalized)]
    paragraphs = [part for part in paragraphs if part]
    if paragraphs:
        return paragraphs

    sentences = re.split(r"(?<=[.!?])\s+", normalize_text(text))
    return [part for part in sentences if part]


def build_chunks(segments: list[str]) -> list[str]:
    if not segments:
        return []

    chunks = []
    current = ""
    for segment in segments:
        candidate = f"{current}\n\n{segment}".strip() if current else segment
        if len(candidate) <= CHUNK_SIZE:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if len(segment) <= CHUNK_SIZE:
            current = segment
            continue

        start = 0
        step = CHUNK_SIZE - CHUNK_OVERLAP
        if step <= 0:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        while start < len(segment):
            piece = segment[start : start + CHUNK_SIZE].strip()
            if piece:
                chunks.append(piece)
            start += step
        current = ""

    if current:
        chunks.append(current)

    return chunks


def read_document(file_path: Path) -> list[dict]:
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return [{"page_number": 1, "text": file_path.read_text(encoding="utf-8")}]

    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        pages = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if normalize_text(text):
                pages.append({"page_number": index, "text": text})
        return pages

    raise ValueError("Unsupported file type. Use a .txt or .pdf file.")


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def ensure_upload_dir() -> Path:
    upload_dir = Path(UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def resolve_input_path(file_path: str) -> Path:
    cleaned = file_path.strip().strip("'\"").strip()
    if not cleaned:
        raise ValueError("File path is empty.")

    path = Path(cleaned).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(WORKSPACE_DIR / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"File not found: {cleaned}")


def ingest_document(file_path: str) -> int:
    path = resolve_input_path(file_path)
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    pages = read_document(path)
    chunk_records = []
    for page in pages:
        page_number = page["page_number"]
        page_text = page["text"]
        segments = split_paragraphs(page_text)
        page_chunks = build_chunks(segments)
        for chunk_text in page_chunks:
            chunk_records.append(
                {
                    "page_number": page_number,
                    "text": chunk_text,
                }
            )

    if not chunk_records:
        raise ValueError("No text could be extracted from the document.")

    model = get_embedding_model()
    chunk_texts = [record["text"] for record in chunk_records]
    embeddings = model.encode(chunk_texts).tolist()
    collection = get_collection()
    filename = path.name
    source_stem = path.stem

    # Remove stale chunks first so repeated ingestion updates the file atomically.
    existing = collection.get(where={"filename": filename})
    if existing.get("ids"):
        collection.delete(ids=existing["ids"])

    ids = [f"{filename}:{index}" for index in range(len(chunk_records))]
    metadatas = [
        {
            "filename": filename,
            "source_stem": source_stem,
            "chunk_index": index,
            "page_number": record["page_number"],
            "file_type": path.suffix.lower().lstrip("."),
        }
        for index, record in enumerate(chunk_records)
    ]

    collection.add(
        ids=ids,
        documents=chunk_texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(chunk_records)


def delete_document(filename: str) -> int:
    cleaned = Path(filename).name.strip()
    if not cleaned:
        raise ValueError("Filename is required.")

    collection = get_collection()
    existing = collection.get(where={"filename": cleaned})
    ids = existing.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def list_documents() -> list[dict]:
    collection = get_collection()
    grouped: dict[str, dict] = {}

    if collection.count() > 0:
        results = collection.get(include=["metadatas"])
        for metadata in results.get("metadatas", []):
            filename = metadata.get("filename", "unknown")
            record = grouped.setdefault(
                filename,
                {
                    "filename": filename,
                    "source_stem": metadata.get("source_stem", ""),
                    "file_type": metadata.get("file_type", ""),
                    "chunk_count": 0,
                    "page_numbers": set(),
                },
            )
            record["chunk_count"] += 1
            page_number = metadata.get("page_number")
            if page_number is not None:
                record["page_numbers"].add(page_number)

    upload_dir = ensure_upload_dir()
    for file_path in upload_dir.iterdir():
        if not file_path.is_file():
            continue
        record = grouped.setdefault(
            file_path.name,
            {
                "filename": file_path.name,
                "source_stem": file_path.stem,
                "file_type": file_path.suffix.lower().lstrip("."),
                "chunk_count": 0,
                "page_numbers": set(),
            },
        )
        record["uploaded_at"] = datetime.fromtimestamp(
            file_path.stat().st_mtime
        ).isoformat()

    documents = []
    for record in grouped.values():
        pages = sorted(record.pop("page_numbers"))
        record["page_count"] = len(pages)
        record["embedded"] = record["chunk_count"] > 0
        record.setdefault("uploaded_at", None)
        documents.append(record)

    documents.sort(key=lambda item: item["filename"].lower())
    return documents


def resolve_uploaded_file(filename: str) -> Path:
    cleaned = Path(filename).name.strip()
    if not cleaned:
        raise ValueError("Filename is required.")

    upload_path = ensure_upload_dir() / cleaned
    if not upload_path.exists():
        raise FileNotFoundError(f"Uploaded file not found: {cleaned}")
    return upload_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a .txt or .pdf file into ChromaDB.")
    parser.add_argument("file_path", help="Path to a .txt or .pdf document")
    args = parser.parse_args()

    try:
        count = ingest_document(args.file_path)
    except Exception as exc:
        print(f"Ingestion failed: {exc}")
        return 1

    print(f"Created and stored {count} chunks from {resolve_input_path(args.file_path).name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
