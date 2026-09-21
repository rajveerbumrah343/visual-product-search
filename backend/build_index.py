"""
One-off / batch script to seed the FAISS index from an existing product
catalog, instead of adding products one-by-one through the API.

Expects:
  1. A folder of product images.
  2. A CSV with columns: image_filename,name,category,price
     (category and price may be left blank per row)

Usage:
    python build_index.py --images-dir ./raw_catalog_images --csv ./catalog.csv

Each image is copied into the served images/ directory (so the API can
serve it later), embedded with the same ResNet50 extractor the live API
uses, and added to the FAISS index in batches for speed.
"""

import argparse
import csv
import os
import shutil
import uuid

from PIL import Image

from config import IMAGES_DIR
from feature_extractor import encode_images_batch
from similarity_search import product_index

BATCH_SIZE = 32


def load_catalog_rows(csv_path: str):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def main(images_dir: str, csv_path: str):
    rows = list(load_catalog_rows(csv_path))
    print(f"Found {len(rows)} rows in {csv_path}")

    batch_images, batch_metas = [], []
    indexed = 0

    for i, row in enumerate(rows, start=1):
        src_path = os.path.join(images_dir, row["image_filename"])
        if not os.path.exists(src_path):
            print(f"  [skip] missing file: {src_path}")
            continue

        # Copy into the served images directory under a unique name
        ext = os.path.splitext(row["image_filename"])[1] or ".jpg"
        stored_filename = f"{uuid.uuid4().hex}{ext}"
        dest_path = os.path.join(IMAGES_DIR, stored_filename)
        shutil.copyfile(src_path, dest_path)

        image = Image.open(dest_path).convert("RGB")
        batch_images.append(image)
        batch_metas.append({
            "product_id": uuid.uuid4().hex,
            "name": row.get("name", "").strip() or "Unnamed product",
            "category": (row.get("category") or "").strip() or None,
            "price": float(row["price"]) if row.get("price") else None,
            "image_path": f"/images/{stored_filename}",
        })

        if len(batch_images) == BATCH_SIZE:
            embeddings = encode_images_batch(batch_images)
            product_index.add_batch(embeddings, batch_metas)
            indexed += len(batch_images)
            print(f"  indexed {indexed}/{len(rows)}")
            batch_images, batch_metas = [], []

    # flush remaining partial batch
    if batch_images:
        embeddings = encode_images_batch(batch_images)
        product_index.add_batch(embeddings, batch_metas)
        indexed += len(batch_images)

    print(f"Done. Indexed {indexed} products. Total in index: {product_index.count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk-index a product catalog for visual search.")
    parser.add_argument("--images-dir", required=True, help="Folder containing raw product images")
    parser.add_argument("--csv", required=True, help="CSV with image_filename,name,category,price columns")
    args = parser.parse_args()
    main(args.images_dir, args.csv)
