"""Build the demo catalogue from Amazon Berkeley Objects (ABO, CC BY 4.0).

    python scripts/build_catalog.py --raw .cache/abo --out data/abo

Selects English listings with a main image from a fixed set of visually
distinct product types, deduplicates near-identical titles, takes up to
``--per-type`` per type (deterministic order: sorted by item_id), downloads
the 256px images, and writes ``products.json``. Nothing is invented: ABO has
no prices or ratings, so the catalogue has none either.

ABO: https://amazon-berkeley-objects.s3.amazonaws.com/index.html
Licence: CC BY 4.0 (c) Amazon.com — see data/abo/ATTRIBUTION.md
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
import time
import urllib.request
from pathlib import Path

BASE = "https://amazon-berkeley-objects.s3.amazonaws.com"
LISTING_SHARDS = [f"listings/metadata/listings_{i}.json.gz" for i in range(2)]
IMAGES_CSV = "images/metadata/images.csv.gz"

# Visually distinct types, so image similarity has something meaningful to find.
TYPES = {
    "SHOES": "Shoes", "BOOT": "Boots", "SANDAL": "Sandals", "HANDBAG": "Handbags",
    "HAT": "Hats", "CHAIR": "Chairs", "SOFA": "Sofas", "TABLE": "Tables",
    "LAMP": "Lamps", "RUG": "Rugs", "WALL_ART": "Wall art", "FINERING": "Rings",
}


def fetch(rel: str, raw: Path) -> Path:
    target = raw / rel.replace("/", "__")
    if not target.exists():
        with urllib.request.urlopen(f"{BASE}/{rel}", timeout=120) as resp:
            target.write_bytes(resp.read())
    return target


def english(field) -> str | None:
    if not field:
        return None
    for preferred in ("en_US", "en_GB", "en_"):
        for item in field:
            if item.get("language_tag", "").startswith(preferred) and item.get("value"):
                return item["value"].strip()
    return None


def all_english(field) -> list[str]:
    return [x["value"].strip() for x in field or [] if x.get("language_tag", "").startswith("en") and x.get("value")]


def dims(item) -> dict | None:
    d = item.get("item_dimensions")
    if not d:
        return None
    try:
        return {k: round(d[k]["normalized_value"]["value"], 2) for k in ("length", "width", "height") if k in d} | {
            "unit": "inches"}
    except (KeyError, TypeError):
        return None


def title_key(title: str) -> str:
    # Variants differ only in size/colour words at the end; compare the first 6 words.
    return " ".join(re.findall(r"[a-z0-9]+", title.lower())[:6])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=".cache/abo")
    p.add_argument("--out", default="data/abo")
    p.add_argument("--per-type", type=int, default=30)
    a = p.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    raw.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(parents=True, exist_ok=True)

    image_paths = {}
    with gzip.open(fetch(IMAGES_CSV, raw), "rt", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            image_paths[row["image_id"]] = row["path"]

    candidates: dict[str, list[dict]] = {t: [] for t in TYPES}
    for shard in LISTING_SHARDS:
        with gzip.open(fetch(shard, raw), "rt", encoding="utf-8") as fh:
            for line in fh:
                item = json.loads(line)
                ptype = (item.get("product_type") or [{}])[0].get("value")
                title = english(item.get("item_name"))
                image_id = item.get("main_image_id")
                if ptype in TYPES and title and image_id in image_paths:
                    candidates[ptype].append(item)

    products, seen_titles = [], set()
    for ptype, items in candidates.items():
        taken = 0
        for item in sorted(items, key=lambda i: i["item_id"]):
            title = english(item["item_name"])
            key = title_key(title)
            if key in seen_titles or taken >= a.per_type:
                continue
            seen_titles.add(key)
            image_file = f"{item['item_id']}.jpg"
            if not (out / "images" / image_file).exists():
                url = f"{BASE}/images/small/{image_paths[item['main_image_id']]}"
                with urllib.request.urlopen(url, timeout=60) as resp:
                    (out / "images" / image_file).write_bytes(resp.read())
                time.sleep(0.05)  # be polite to the public bucket
            weight = (item.get("item_weight") or [{}])[0].get("normalized_value")
            products.append({
                "id": item["item_id"],
                "title": title,
                "brand": english(item.get("brand")) or "Unbranded",
                "category": TYPES[ptype],
                "color": english(item.get("color")),
                "material": english(item.get("material")),
                "style": english(item.get("style")),
                "features": all_english(item.get("bullet_point"))[:6],
                "keywords": all_english(item.get("item_keywords"))[:15],
                "dimensions": dims(item),
                "weight_lb": round(weight["value"], 2) if weight and weight.get("unit") == "pounds" else None,
                "image": image_file,
                "source": f"ABO listing {item['item_id']}",
            })
            taken += 1
        print(f"{ptype:10s} {taken:3d} of {len(items)} candidates")

    (out / "products.json").write_text(json.dumps(products, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(products)} products to {out / 'products.json'}")


if __name__ == "__main__":
    main()
