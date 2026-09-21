
import os
import uuid
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import DEFAULT_TOP_K, IMAGES_DIR
from feature_extractor import bytes_to_image, encode_image
from similarity_search import product_index

app = FastAPI(
    title="Visual Product Similarity & Recommendation API",
    description="Finds visually similar products from an uploaded image using "
                "deep-learning embeddings and FAISS similarity search.",
    version="1.0.0",
)

# Allow the React dev server (and any frontend) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten this to your frontend's origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded/stored product images directly, e.g. GET /images/<file>.jpg
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


# ---------- response/request models ----------

class ProductOut(BaseModel):
    product_id: str
    name: str
    category: Optional[str] = None
    price: Optional[float] = None
    image_path: str


class SearchResultOut(ProductOut):
    similarity: float


class SearchResponse(BaseModel):
    query_time_ms: float
    results: List[SearchResultOut]


# ---------- helpers ----------

def _save_upload_to_disk(file_bytes: bytes, original_filename: str) -> str:
    """Persist an uploaded image under a unique name; return the stored filename."""
    ext = os.path.splitext(original_filename)[1] or ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(IMAGES_DIR, filename)
    with open(path, "wb") as f:
        f.write(file_bytes)
    return filename


# ---------- endpoints ----------

@app.get("/api/health")
def health():
    return {"status": "ok", "indexed_products": product_index.count()}


@app.post("/api/search", response_model=SearchResponse)
async def search_similar_products(
    file: UploadFile = File(...),
    top_k: int = DEFAULT_TOP_K,
):
    """
    Core recommendation endpoint: upload a query image, get back the
    top_k most visually similar products already in the catalog.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    import time
    start = time.perf_counter()

    raw_bytes = await file.read()
    try:
        image = bytes_to_image(raw_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image file.")

    query_embedding = encode_image(image)
    results = product_index.search(query_embedding, top_k=top_k)

    elapsed_ms = (time.perf_counter() - start) * 1000
    return SearchResponse(query_time_ms=round(elapsed_ms, 2), results=results)


@app.post("/api/products", response_model=ProductOut)
async def add_product(
    file: UploadFile = File(...),
    name: str = Form(...),
    category: Optional[str] = Form(None),
    price: Optional[float] = Form(None),
):
    """Add a new product to the catalog: stores the image, extracts its
    embedding, and indexes it for future similarity search."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    raw_bytes = await file.read()
    try:
        image = bytes_to_image(raw_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image file.")

    stored_filename = _save_upload_to_disk(raw_bytes, file.filename)
    embedding = encode_image(image)

    product_id = uuid.uuid4().hex
    meta = {
        "product_id": product_id,
        "name": name,
        "category": category,
        "price": price,
        "image_path": f"/images/{stored_filename}",
    }
    product_index.add(embedding, meta)

    return ProductOut(**meta)


@app.get("/api/products", response_model=List[ProductOut])
def list_products():
    return [ProductOut(**m) for m in product_index.metadata]


@app.get("/api/products/{product_id}", response_model=ProductOut)
def get_product(product_id: str):
    product = product_index.get_product(product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found.")
    return ProductOut(**product)
