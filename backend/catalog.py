"""Product catalogue over the Amazon Berkeley Objects subset (data/abo, CC BY 4.0).

Keyword search is BM25 over title, brand, category, colour, material, style,
features and keywords. Everything here is pure Python and loads in
milliseconds, so it runs (and is tested) without any ML model. Vector search
lives in backend/vectors.py and is combined with this ranking in hybrid mode.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ABO_DIR = Path(os.getenv("PVA_ABO_DIR", Path(__file__).resolve().parent.parent / "data" / "abo"))

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset("a an and for in of on or the to with by".split())
# Field weights: a match in the title says more than one in a keyword list.
FIELD_WEIGHTS = {"title": 3, "brand": 2, "category": 2, "color": 2, "style": 2, "material": 1, "features": 1, "keywords": 1}
FACETS = ("category", "brand", "color")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def _clean_keywords(raw: list[str]) -> list[str]:
    # ABO keyword lists repeat themselves and sometimes pack many phrases into
    # one "^"-separated string; split and de-duplicate, keeping order.
    seen: dict[str, None] = {}
    for item in raw:
        for kw in item.split("^"):
            kw = kw.strip()
            if kw:
                seen.setdefault(kw.lower(), None)
    return list(seen)


@dataclass(frozen=True)
class Product:
    id: str
    title: str
    brand: str | None
    category: str
    color: str | None
    material: str | None
    style: str | None
    features: tuple[str, ...]
    keywords: tuple[str, ...]
    image: str
    source: str

    def public(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "brand": self.brand,
            "category": self.category,
            "color": self.color,
            "material": self.material,
            "style": self.style,
            "features": list(self.features),
            "image_url": f"/images/{self.image}",
            "source": self.source,
        }

    def field_text(self, name: str) -> str:
        value = getattr(self, name)
        if isinstance(value, tuple):
            return " ".join(value)
        return value or ""


@dataclass
class Catalog:
    products: dict[str, Product]
    k1: float = 1.2
    b: float = 0.75
    _tf: dict[str, Counter] = field(default_factory=dict, repr=False)
    _len: dict[str, float] = field(default_factory=dict, repr=False)
    _df: Counter = field(default_factory=Counter, repr=False)
    _avg_len: float = 1.0

    def __post_init__(self) -> None:
        for pid, p in self.products.items():
            tf: Counter = Counter()
            for name, weight in FIELD_WEIGHTS.items():
                for tok in tokenize(p.field_text(name)):
                    tf[tok] += weight
            self._tf[pid] = tf
            self._len[pid] = sum(tf.values())
            self._df.update(tf.keys())
        if self.products:
            self._avg_len = sum(self._len.values()) / len(self.products)

    # ---------- lookup ----------

    def get(self, product_id: str) -> Product | None:
        return self.products.get(product_id)

    def __len__(self) -> int:
        return len(self.products)

    # ---------- ranking ----------

    def bm25(self, query: str) -> list[tuple[str, float]]:
        """(product_id, score) for every product matching at least one query term, best first."""
        terms = tokenize(query)
        if not terms:
            return []
        n = len(self.products)
        scores: dict[str, float] = {}
        for term in set(terms):
            df = self._df.get(term)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for pid, tf in self._tf.items():
                f = tf.get(term)
                if not f:
                    continue
                norm = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self._len[pid] / self._avg_len))
                scores[pid] = scores.get(pid, 0.0) + idf * norm
        # Ties broken by id so results are deterministic.
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))

    # ---------- filtering ----------

    def matches(self, p: Product, filters: dict[str, str | None]) -> bool:
        for name, wanted in filters.items():
            if wanted and (getattr(p, name) or "").lower() != wanted.lower():
                return False
        return True

    def facets(self, ids: list[str] | None = None) -> dict[str, list[dict]]:
        pool = [self.products[i] for i in ids] if ids is not None else list(self.products.values())
        out = {}
        for name in FACETS:
            counts = Counter(getattr(p, name) for p in pool if getattr(p, name))
            out[name] = [{"value": v, "count": c} for v, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        return out


def load_catalog(abo_dir: Path = ABO_DIR) -> Catalog:
    rows = json.loads((abo_dir / "products.json").read_text(encoding="utf-8"))
    products = {}
    for r in rows:
        products[r["id"]] = Product(
            id=r["id"],
            title=r["title"],
            brand=r.get("brand"),
            category=r["category"],
            color=r.get("color"),
            material=r.get("material"),
            style=r.get("style"),
            features=tuple(r.get("features") or ()),
            keywords=tuple(_clean_keywords(r.get("keywords") or [])),
            image=r["image"],
            source=r["source"],
        )
    return Catalog(products)


@lru_cache(maxsize=1)
def default_catalog() -> Catalog:
    return load_catalog()
