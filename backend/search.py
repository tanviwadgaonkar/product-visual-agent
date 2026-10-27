"""Search over the catalogue: keyword, semantic (CLIP) or hybrid, plus filters, sorting and paging."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from backend import vectors
from backend.catalog import Catalog

Mode = Literal["auto", "keyword", "semantic", "hybrid"]
Sort = Literal["relevance", "title_asc", "title_desc"]

# Dense retrieval ranks *every* product; only the head is meaningful evidence
# of relevance, so fusion and semantic results use the top N.
DENSE_TOP_N = 40


class SearchUnavailable(RuntimeError):
    """Semantic search was requested but no CLIP encoder / vectors are available."""


@dataclass
class SearchResult:
    items: list[dict]
    total: int
    page: int
    page_size: int
    mode: str
    facets: dict


def search(
    catalog: Catalog,
    q: str = "",
    *,
    category: str | None = None,
    brand: str | None = None,
    color: str | None = None,
    mode: Mode = "auto",
    sort: Sort = "relevance",
    page: int = 1,
    page_size: int = 24,
    index: vectors.VectorIndex | None = None,
    encoder: vectors.Encoder | None = None,
) -> SearchResult:
    q = q.strip()
    semantic_ok = index is not None and encoder is not None

    if not q:
        used = "browse"
        ranked = [(pid, 0.0) for pid in sorted(catalog.products, key=lambda i: catalog.products[i].title.lower())]
    else:
        if mode == "auto":
            mode = "hybrid" if semantic_ok else "keyword"
        if mode in ("semantic", "hybrid") and not semantic_ok:
            raise SearchUnavailable("semantic search needs the CLIP encoder (PVA_CLIP=1) and data/abo/embeddings.npz")
        used = mode
        if mode == "keyword":
            ranked = catalog.bm25(q)
        else:
            dense = index.by_text_vector(encoder.encode_text(q))[:DENSE_TOP_N]
            ranked = dense if mode == "semantic" else vectors.rrf(catalog.bm25(q), dense)

    filters = {"category": category, "brand": brand, "color": color}
    hits = [(pid, s) for pid, s in ranked if pid in catalog.products and catalog.matches(catalog.products[pid], filters)]

    if sort != "relevance":
        hits.sort(key=lambda h: catalog.products[h[0]].title.lower(), reverse=sort == "title_desc")

    start = (page - 1) * page_size
    items = [{**catalog.products[pid].public(), "score": round(s, 4)} for pid, s in hits[start:start + page_size]]
    return SearchResult(items=items, total=len(hits), page=page, page_size=page_size, mode=used,
                        facets=catalog.facets([pid for pid, _ in hits]))


COMPARE_FIELDS = ("brand", "category", "color", "material", "style")


def compare(catalog: Catalog, ids: list[str]) -> dict:
    """Side-by-side attributes, and which of them the products share or differ on."""
    products = [catalog.products[i] for i in ids]
    rows = []
    for name in COMPARE_FIELDS:
        values = [getattr(p, name) for p in products]
        known = {v.lower() for v in values if v}
        rows.append({"field": name, "values": values, "same": len(known) == 1 and all(values)})
    features = [set(p.features) for p in products]
    return {
        "products": [p.public() for p in products],
        "attributes": rows,
        "shared_features": sorted(set.intersection(*features)) if features else [],
    }
