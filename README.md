# Visual Product Similarity & Image-Based Recommendation System

Find and recommend visually similar products from a product image, using a
pretrained CNN for feature extraction and FAISS for similarity search.

## How it works

```
Uploaded image
      │
      ▼
Preprocess (resize, normalize)  ──► feature_extractor.py
      │
      ▼
ResNet50 (ImageNet, fc layer removed)
      │
      ▼
2048-dim embedding (L2-normalized)
      │
      ▼
FAISS IndexFlatIP (inner product = cosine similarity) ──► similarity_search.py
      │
      ▼
Top-K most similar products (id, name, category, price, image, score)
      │
      ▼
React UI renders results, sorted by similarity
```

- **Feature extraction**: ResNet50 pretrained on ImageNet, with the final
  classification layer stripped off, so the model outputs a 2048-dim pooled
  feature vector per image instead of a class label. This vector is the
  product's "visual embedding."
- **Similarity search**: embeddings are L2-normalized, so inner product ==
  cosine similarity. FAISS's `IndexFlatIP` does exact nearest-neighbor
  search over all indexed products.
- **Metadata**: kept in a simple parallel JSON file (`data/index/metadata.json`),
  indexed by the same position as the FAISS vectors.

## Project structure

```
backend/
  main.py              FastAPI app (upload, search, add-product endpoints)
  feature_extractor.py ResNet50-based embedding extraction
  similarity_search.py FAISS index + metadata wrapper
  build_index.py        Bulk-index an existing catalog from images + CSV
  config.py             Paths and constants
  requirements.txt
  data/
    images/             Stored product images (served at /images/<file>)
    index/               FAISS index + metadata.json (created at runtime)

frontend/
  src/App.jsx           Upload UI + results grid
  src/main.jsx
  index.html
  package.json
  vite.config.js
```

## Setup

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The API is now live at `http://localhost:8000`. Interactive docs (Swagger UI)
are auto-generated at `http://localhost:8000/docs`.

### 2. Seed the catalog

You need some products in the index before search is useful. Two options:

**Option A — one at a time, via the API:**
```bash
curl -X POST http://localhost:8000/api/products \
  -F "file=@/path/to/shoe.jpg" \
  -F "name=Running Shoe - Blue" \
  -F "category=Footwear" \
  -F "price=59.99"
```

**Option B — bulk import from a folder + CSV:**

`catalog.csv`:
```csv
image_filename,name,category,price
shoe1.jpg,Running Shoe - Blue,Footwear,59.99
shoe2.jpg,Running Shoe - Red,Footwear,64.99
bag1.jpg,Leather Tote Bag,Bags,89.00
```

```bash
python build_index.py --images-dir ./raw_catalog_images --csv ./catalog.csv
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Uploads are proxied to the backend automatically
(see `vite.config.js`).

## API reference

| Method | Endpoint              | Description                                |
|--------|------------------------|---------------------------------------------|
| POST   | `/api/search`          | Upload an image, get top-K similar products |
| POST   | `/api/products`        | Add a new product (image + metadata)        |
| GET    | `/api/products`        | List all indexed products                   |
| GET    | `/api/products/{id}`   | Fetch one product's metadata                |
| GET    | `/api/health`          | Health check + index size                   |

## Notes on scaling this up

- **Bigger catalogs**: `IndexFlatIP` is exact but O(n) per query. Past ~1M
  products, switch to `IndexIVFFlat` or `IndexHNSWFlat` for approximate
  nearest-neighbor search with sub-linear query time.
- **Better embeddings**: ResNet50 gives strong generic visual features. For
  more semantic similarity (e.g. matching by product *type* rather than pure
  visual texture/color), swap in CLIP's image encoder — `encode_image()` in
  `feature_extractor.py` is the only place that needs to change.
- **Persistence**: metadata currently lives in a JSON file for simplicity.
  For a real production catalog, move it to Postgres/SQLite and keep FAISS
  purely for vector search, joining on `product_id`.
- **GPU**: the code auto-detects CUDA (`torch.cuda.is_available()`) and will
  use it if present — no code changes needed.
