"""
Central configuration for the Visual Product Similarity backend.
Keeping all paths/constants in one place makes the rest of the
codebase easy to reason about and easy to change (e.g. swapping
the embedding model or moving the index to disk elsewhere).
"""

import os

# ---------- Directories ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")          # stored product images
INDEX_DIR = os.path.join(DATA_DIR, "index")            # FAISS index + metadata

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "product.index")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")

# ---------- Model ----------
# 2048-dim embeddings from ResNet50 (ImageNet pretrained), global-average-pooled,
# with the final classification layer removed.
EMBEDDING_DIM = 2048
IMAGE_SIZE = 224

# ---------- Search ----------
DEFAULT_TOP_K = 8
