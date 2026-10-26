"""Ingestion and retrieval over a local ChromaDB collection.

Embeddings come from Ollama (nomic-embed-text by default), so nothing leaves the machine.
"""
import csv
import io
import json
import os
import tempfile
from pathlib import Path

import ollama

DATA_DIR = Path(os.getenv("PVA_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
CHROMA_DIR = DATA_DIR / "chroma"
PRODUCTS_JSON = DATA_DIR / "products.json"
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
COLLECTION = "kite_catalogue"

CSV_COLUMNS = ("product_id", "name", "category", "colours", "sizes", "price_inr", "description", "features")
TEXT_SUFFIXES = (".md", ".txt", ".pdf")


class IngestError(ValueError):
    """The uploaded file can't be ingested (bad type, missing columns, ...)."""


_collection = None


def collection():
    """The Chroma collection, created on first use (importing this module touches no disk)."""
    global _collection
    if _collection is None:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    return _collection


def set_collection(coll) -> None:
    """Swap the vector store (tests use an in-memory stand-in)."""
    global _collection
    _collection = coll


# ---------- embeddings ----------

def embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    out = ollama.embed(model=EMBED_MODEL, input=texts)
    return list(out["embeddings"])


# ---------- products ----------

def load_products() -> dict[str, dict]:
    if PRODUCTS_JSON.exists():
        return json.loads(PRODUCTS_JSON.read_text(encoding="utf-8"))
    return {}


def _save_products(products: dict[str, dict]) -> None:
    # Write-then-rename so a crash never leaves a half-written catalogue.
    PRODUCTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=PRODUCTS_JSON.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(products, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, PRODUCTS_JSON)


def _product_text(p: dict) -> str:
    return (
        f"Product: {p['name']} (ID {p['product_id']})\n"
        f"Category: {p['category']}\n"
        f"Colours: {p['colours']}\n"
        f"Sizes: {p['sizes']}\n"
        f"Price: ₹{p['price_inr']}\n"
        f"Description: {p['description']}\n"
        f"Features: {p['features']}"
    )


def _parse_catalogue(raw: bytes) -> list[dict]:
    """Validate a catalogue CSV fully before anything is written."""
    try:
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
        header = [h.strip() for h in (reader.fieldnames or [])]
        rows = [{(k or "").strip(): (v or "").strip() for k, v in r.items() if k} for r in reader]
    except (UnicodeDecodeError, csv.Error) as exc:
        raise IngestError(f"could not read CSV: {exc}") from exc

    missing = [c for c in CSV_COLUMNS if c not in header]
    if missing:
        raise IngestError(f"CSV is missing columns: {', '.join(missing)}")
    seen: set[str] = set()
    for line, r in enumerate(rows, start=2):
        pid = r["product_id"]
        if not pid:
            raise IngestError(f"row {line}: product_id is empty")
        if pid in seen:
            raise IngestError(f"row {line}: duplicate product_id {pid}")
        seen.add(pid)
    return rows


# ---------- chunking ----------

def _chunk(text: str, size: int = 700, overlap: int = 120) -> list[str]:
    text = text.replace("\r\n", "\n")
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        # try to break on a paragraph or sentence boundary
        cut = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
        if cut <= start + size // 2:
            cut = end
        else:
            cut += 1
        chunks.append(text[start:cut].strip())
        if cut >= len(text):
            break
        start = max(cut - overlap, start + 1)
        # never start a chunk mid-word
        while start < len(text) and not text[start - 1].isspace():
            start += 1
    return [c for c in chunks if c]


def _read_pdf(raw: bytes) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(raw))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    except (PdfReadError, ValueError) as exc:
        raise IngestError(f"could not read PDF: {exc}") from exc


# ---------- ingestion ----------

def ingest_bytes(filename: str, raw: bytes) -> dict:
    """Ingest a CSV catalogue or a text/markdown/PDF document. Returns counts.

    Re-uploading a file with the same name replaces it. Order matters: the
    file is parsed and embedded *before* the previous version is removed, so a
    bad file or an Ollama outage leaves the existing index untouched.
    """
    name = filename.lower()
    ids, docs, metas = [], [], []
    catalogue_rows: list[dict] | None = None

    if name.endswith(".csv"):
        catalogue_rows = _parse_catalogue(raw)
        for r in catalogue_rows:
            ids.append(f"{filename}::{r['product_id']}")
            docs.append(_product_text(r))
            metas.append({"source": filename, "product_id": r["product_id"], "kind": "product"})
    elif name.endswith(TEXT_SUFFIXES):
        text = _read_pdf(raw) if name.endswith(".pdf") else raw.decode("utf-8", errors="ignore")
        for i, chunk in enumerate(_chunk(text)):
            ids.append(f"{filename}::chunk{i}")
            docs.append(chunk)
            metas.append({"source": filename, "product_id": "", "kind": "doc", "chunk": i})
    else:
        raise IngestError("unsupported file type: upload .csv, .md, .txt or .pdf")

    if not docs:
        return {"file": filename, "chunks": 0}

    embeddings = embed(docs)  # may raise; nothing has been modified yet

    coll = collection()
    old = coll.get(where={"source": filename}, include=[])["ids"]
    if old:
        coll.delete(ids=old)
    coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

    if catalogue_rows is not None:
        # Products from this file are replaced as a set, so a product removed
        # from the CSV also disappears from the product cards.
        products = {pid: p for pid, p in load_products().items() if p.get("_source") != filename}
        for r in catalogue_rows:
            products[r["product_id"]] = {**r, "_source": filename}
        _save_products(products)

    return {"file": filename, "chunks": len(docs)}


def ingest_path(path: Path) -> dict:
    return ingest_bytes(path.name, path.read_bytes())


def count() -> int:
    return collection().count()


# ---------- retrieval ----------

def retrieve(question: str, k: int = 4) -> list[dict]:
    total = count()
    if total == 0:
        return []
    res = collection().query(query_embeddings=embed([question]), n_results=min(k, total))
    out = []
    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i] or {}
        out.append(
            {
                "id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "source": meta.get("source", ""),
                "product_id": meta.get("product_id") or None,
                "score": round(1 - (res["distances"][0][i] if res.get("distances") else 0), 3),
            }
        )
    return out
