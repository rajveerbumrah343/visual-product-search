

import os

from PIL import Image, UnidentifiedImageError

from config import IMAGES_DIR
from feature_extractor import encode_images_batch
from similarity_search import product_index

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
BATCH_SIZE = 32


def clean_name(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    return stem.replace("_", " ").replace("-", " ").strip().title()


def already_indexed_paths() -> set:
    """Avoid re-indexing images that were already added in a previous run."""
    return {item["image_path"] for item in product_index.metadata}


def find_images(root: str):
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in sorted(filenames):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in VALID_EXTS:
                continue
            full_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(full_path, root)
            category = None
            parent = os.path.dirname(rel_path)
            if parent and parent != ".":
                # use the first-level subfolder as category, e.g. shoes/nike/x.jpg -> "shoes"
                category = parent.split(os.sep)[0]
            yield full_path, rel_path.replace(os.sep, "/"), fname, category


def main():
    already = already_indexed_paths()

    batch_images, batch_metas = [], []
    indexed, skipped = 0, 0

    for full_path, rel_path, fname, category in find_images(IMAGES_DIR):
        url_path = f"/images/{rel_path}"
        if url_path in already:
            skipped += 1
            continue

        try:
            image = Image.open(full_path).convert("RGB")
        except UnidentifiedImageError:
            print(f"  [skip] not a valid image: {full_path}")
            continue

        import uuid
        batch_images.append(image)
        batch_metas.append({
            "product_id": uuid.uuid4().hex,
            "name": clean_name(fname),
            "category": category,
            "price": None,
            "image_path": url_path,
        })

        if len(batch_images) == BATCH_SIZE:
            embeddings = encode_images_batch(batch_images)
            product_index.add_batch(embeddings, batch_metas)
            indexed += len(batch_images)
            print(f"  indexed {indexed} so far...")
            batch_images, batch_metas = [], []

    if batch_images:
        embeddings = encode_images_batch(batch_images)
        product_index.add_batch(embeddings, batch_metas)
        indexed += len(batch_images)

    print(f"\nDone. Newly indexed: {indexed}. Already indexed (skipped): {skipped}.")
    print(f"Total products in index: {product_index.count()}")


if __name__ == "__main__":
    main()
