"""
Vector index wrapper around FAISS.

Design choices:
- IndexFlatIP (inner product) over normalized embeddings == cosine similarity.
  Flat = exact search (no approximation). Fine up to ~1-5M vectors; beyond
  that you'd switch to IndexIVFFlat/IndexHNSWFlat for speed.
- Metadata (product id, name, category, price, image path) is kept in a
  parallel JSON list, indexed by the same integer position as the FAISS
  vectors. This keeps FAISS doing only what it's good at (vector math)
  while metadata stays simple and human-readable.
- The whole thing is wrapped in a class with a lock so the FastAPI app can
  safely add products while search requests are in flight.
"""

import json
import os
import threading
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from config import EMBEDDING_DIM, FAISS_INDEX_PATH, METADATA_PATH


class ProductIndex:
    def __init__(self):
        self._lock = threading.RLock()
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.metadata: List[Dict[str, Any]] = []
        self._load()

    # ---------- persistence ----------
    def _load(self) -> None:
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            with open(METADATA_PATH, "r") as f:
                self.metadata = json.load(f)

    def save(self) -> None:
        with self._lock:
            faiss.write_index(self.index, FAISS_INDEX_PATH)
            with open(METADATA_PATH, "w") as f:
                json.dump(self.metadata, f, indent=2)

    # ---------- mutation ----------
    def add(self, embedding: np.ndarray, meta: Dict[str, Any]) -> int:
        """Add a single product embedding + metadata. Returns its position id."""
        with self._lock:
            vector = embedding.reshape(1, -1).astype("float32")
            self.index.add(vector)
            self.metadata.append(meta)
            position_id = len(self.metadata) - 1
            self.save()
            return position_id

    def add_batch(self, embeddings: np.ndarray, metas: List[Dict[str, Any]]) -> None:
        with self._lock:
            self.index.add(embeddings.astype("float32"))
            self.metadata.extend(metas)
            self.save()

    # ---------- search ----------
    def search(self, query_embedding: np.ndarray, top_k: int = 8) -> List[Dict[str, Any]]:
        with self._lock:
            if self.index.ntotal == 0:
                return []

            top_k = min(top_k, self.index.ntotal)
            query = query_embedding.reshape(1, -1).astype("float32")
            scores, indices = self.index.search(query, top_k)  # inner product = cosine sim

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                item = dict(self.metadata[idx])
                item["similarity"] = float(score)  # 1.0 = identical, 0 = unrelated, -1 = opposite
                results.append(item)
            return results

    def count(self) -> int:
        return self.index.ntotal

    def get_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        for item in self.metadata:
            if item.get("product_id") == product_id:
                return item
        return None


# Singleton used by the FastAPI app
product_index = ProductIndex()
