"""CLIP vectors for the catalogue, and hybrid ranking.

`data/abo/embeddings.npz` holds L2-normalised CLIP ViT-B/32 vectors for every
product image and title. It is produced by scripts/embed_catalog.py (run in
GitHub Actions, not on a laptop) and committed, so "visually similar
products" needs only numpy at runtime.

Encoding *new* input (a free-text query, an uploaded photo) needs the CLIP
model itself. That is optional: set PVA_CLIP=1 and install
requirements-ml.txt. Without it, search falls back to keyword-only and the
API says so.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from backend.catalog import ABO_DIR

CLIP_MODEL = os.getenv("PVA_CLIP_MODEL", "clip-ViT-B-32")
RRF_K = 60


@dataclass
class VectorIndex:
    ids: list[str]
    image: np.ndarray  # (n, d), rows L2-normalised
    text: np.ndarray  # (n, d)
    model: str

    def __post_init__(self) -> None:
        self._row = {pid: i for i, pid in enumerate(self.ids)}

    def __contains__(self, product_id: str) -> bool:
        return product_id in self._row

    def _rank(self, matrix: np.ndarray, query: np.ndarray, exclude: str | None = None) -> list[tuple[str, float]]:
        q = query / (np.linalg.norm(query) or 1.0)
        sims = matrix @ q
        order = np.argsort(-sims, kind="stable")
        return [(self.ids[i], float(sims[i])) for i in order if self.ids[i] != exclude]

    def similar_to(self, product_id: str) -> list[tuple[str, float]]:
        """Products whose *images* look most like this product's image."""
        return self._rank(self.image, self.image[self._row[product_id]], exclude=product_id)

    def by_text_vector(self, vec: np.ndarray) -> list[tuple[str, float]]:
        # A text query is compared against images (cross-modal, CLIP's strength)
        # and titles; averaging both is more robust than either alone.
        return self._rank((self.image + self.text) / 2, vec)

    def by_image_vector(self, vec: np.ndarray) -> list[tuple[str, float]]:
        return self._rank(self.image, vec)


def load_index(path: Path | None = None) -> VectorIndex | None:
    path = path or ABO_DIR / "embeddings.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        return VectorIndex(ids=[str(i) for i in z["ids"]], image=z["image"].astype(np.float32),
                           text=z["text"].astype(np.float32), model=str(z["model"]))


@lru_cache(maxsize=1)
def default_index() -> VectorIndex | None:
    return load_index()


# ---------- query encoder (optional, heavy) ----------

class Encoder:
    """Encodes text or images into the same CLIP space as the stored vectors."""

    def encode_text(self, text: str) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def encode_image(self, raw: bytes) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class ClipEncoder(Encoder):  # pragma: no cover - needs torch; exercised in the embeddings workflow
    def __init__(self, model_name: str = CLIP_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")

    def encode_text(self, text: str) -> np.ndarray:
        return self.model.encode([text], normalize_embeddings=True)[0]

    def encode_image(self, raw: bytes) -> np.ndarray:
        from PIL import Image

        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return self.model.encode([img], normalize_embeddings=True)[0]


_encoder: Encoder | None = None
_encoder_loaded = False


def encoder() -> Encoder | None:
    """The CLIP encoder if enabled (PVA_CLIP=1), loaded once on first use."""
    global _encoder, _encoder_loaded
    if not _encoder_loaded:
        _encoder_loaded = True
        if os.getenv("PVA_CLIP") == "1":
            _encoder = ClipEncoder()
    return _encoder


def set_encoder(enc: Encoder | None) -> None:
    """Swap the encoder (tests use a deterministic fake)."""
    global _encoder, _encoder_loaded
    _encoder, _encoder_loaded = enc, True


# ---------- fusion ----------

def rrf(*rankings: list[tuple[str, float]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: robust to the two rankers' scores being on different scales."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (pid, _) in enumerate(ranking, start=1):
            fused[pid] = fused.get(pid, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
