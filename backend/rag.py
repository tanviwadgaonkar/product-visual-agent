"""Ingestion and retrieval over a local ChromaDB collection.

Embeddings come from Ollama (nomic-embed-text by default), so nothing leaves the machine.
"""
import csv
import io
import json
import os
from pathlib import Path

import chromadb
import ollama

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_DIR = DATA_DIR / "chroma"
PRODUCTS_JSON = DATA_DIR / "products.json"
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
COLLECTION = "kite_catalogue"

_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


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
    PRODUCTS_JSON.write_text(json.dumps(products, ensure_ascii=False, indent=2), encoding="utf-8")


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
    reader = PdfReader(io.BytesIO(raw))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


# ---------- ingestion ----------

def ingest_bytes(filename: str, raw: bytes) -> dict:
    """Ingest a CSV catalogue or a text/markdown/PDF document. Returns counts."""
    name = filename.lower()
    ids, docs, metas = [], [], []

    if name.endswith(".csv"):
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        products = load_products()
        for r in rows:
            r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
            pid = r.get("product_id") or f"P-{len(products) + 1:03d}"
            r["product_id"] = pid
            products[pid] = r
            ids.append(f"{filename}::{pid}")
            docs.append(_product_text(r))
            metas.append({"source": filename, "product_id": pid, "kind": "product"})
        _save_products(products)
    else:
        text = _read_pdf(raw) if name.endswith(".pdf") else raw.decode("utf-8", errors="ignore")
        for i, chunk in enumerate(_chunk(text)):
            ids.append(f"{filename}::chunk{i}")
            docs.append(chunk)
            metas.append({"source": filename, "product_id": "", "kind": "doc", "chunk": i})

    if not docs:
        return {"file": filename, "chunks": 0}

    # replace any previous version of this file
    try:
        _collection.delete(where={"source": filename})
    except Exception:
        pass
    _collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embed(docs))
    return {"file": filename, "chunks": len(docs)}


def ingest_path(path: Path) -> dict:
    return ingest_bytes(path.name, path.read_bytes())


def count() -> int:
    return _collection.count()


# ---------- retrieval ----------

def retrieve(question: str, k: int = 4) -> list[dict]:
    if count() == 0:
        return []
    res = _collection.query(query_embeddings=embed([question]), n_results=min(k, count()))
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
