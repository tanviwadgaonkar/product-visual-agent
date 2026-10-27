# Data attribution

`products.json` and `images/` are a 360-product subset of **Amazon Berkeley Objects (ABO)**, (c) Amazon.com, licensed under **CC BY 4.0** (`LICENSE-CC-BY-4.0.txt`).

- Source: https://amazon-berkeley-objects.s3.amazonaws.com/index.html
- Citation: Collins et al., *ABO: Dataset and Benchmarks for Real-World 3D Object Understanding*, CVPR 2022.
- Changes: selected 30 English listings with a main image from each of 12 product types, deduplicated by title, reduced fields, 256px images. Built by `scripts/build_catalog.py`.

ABO has no prices or ratings, so this catalogue has none.
