"""Compute CLIP vectors for the ABO catalogue and evaluate them.

Heavy (downloads ~600 MB of model weights, runs torch). Run it in GitHub
Actions (.github/workflows/embeddings.yml), not on a laptop:

    pip install -r requirements-ml.txt
    python scripts/embed_catalog.py            # writes data/abo/embeddings.npz + data/abo/embedding_eval.json

Evaluation uses only labels that are in the data (product category), so the
numbers are reproducible and nothing is hand-labelled:
  - image->image: of each product's 5 nearest images, the share in the same category
  - text->image:  for the query "a photo of <category>", the share of the top 10 in that category
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.catalog import ABO_DIR, load_catalog  # noqa: E402


def precision_at_k(ranked_categories: list[list[str]], targets: list[str], k: int) -> float:
    return float(np.mean([sum(c == t for c in cats[:k]) / k for cats, t in zip(ranked_categories, targets)]))


def evaluate(ids: list[str], image: np.ndarray, categories: dict[str, str], encode_text) -> dict:
    cats = [categories[i] for i in ids]
    sims = image @ image.T
    np.fill_diagonal(sims, -np.inf)
    nn = np.argsort(-sims, axis=1)[:, :5]
    i2i = precision_at_k([[cats[j] for j in row] for row in nn], cats, 5)

    labels = sorted(set(cats))
    queries = encode_text([f"a photo of {c.lower()}" for c in labels])
    top = np.argsort(-(queries @ image.T), axis=1)[:, :10]
    per_label = {c: sum(cats[j] == c for j in row) / 10 for c, row in zip(labels, top)}
    return {
        "products": len(ids),
        "categories": len(labels),
        "image_to_image_precision_at_5": round(i2i, 4),
        "text_to_image_precision_at_10": round(float(np.mean(list(per_label.values()))), 4),
        "text_to_image_precision_at_10_by_category": {c: round(v, 2) for c, v in per_label.items()},
        "chance_level": round(1 / len(labels), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="clip-ViT-B-32")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    from PIL import Image
    from sentence_transformers import SentenceTransformer

    catalog = load_catalog()
    ids = sorted(catalog.products)
    model = SentenceTransformer(args.model, device="cpu")

    def encode_text(texts):
        return model.encode(texts, batch_size=args.batch, normalize_embeddings=True, convert_to_numpy=True)

    started = time.perf_counter()
    images = [Image.open(ABO_DIR / "images" / catalog.products[i].image).convert("RGB") for i in ids]
    image = model.encode(images, batch_size=args.batch, normalize_embeddings=True, convert_to_numpy=True)
    # CLIP's text tower truncates at 77 tokens; titles fit, keyword lists would not.
    text = encode_text([f"{catalog.products[i].title}. {catalog.products[i].category}" for i in ids])
    seconds = time.perf_counter() - started

    out = ABO_DIR / "embeddings.npz"
    np.savez_compressed(out, ids=np.array(ids), image=image.astype(np.float16), text=text.astype(np.float16),
                        model=np.array(args.model))

    report = evaluate(ids, image, {i: catalog.products[i].category for i in ids}, encode_text)
    report.update(model=args.model, dim=int(image.shape[1]), encode_seconds=round(seconds, 1))
    (ABO_DIR / "embedding_eval.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
