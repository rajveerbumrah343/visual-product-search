

import io
import threading

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from config import IMAGE_SIZE

_lock = threading.Lock()
_model = None
_transform = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model() -> nn.Module:
    """Lazily load ResNet50 once per process, with the classifier head removed."""
    global _model, _transform
    if _model is not None:
        return _model

    with _lock:
        if _model is not None:  # double-checked locking
            return _model

        weights = models.ResNet50_Weights.IMAGENET1K_V2
        backbone = models.resnet50(weights=weights)
        # Drop the final `fc` layer -> output is the (B, 2048, 1, 1) pooled feature map
        modules = list(backbone.children())[:-1]
        model = nn.Sequential(*modules)
        model.eval()
        model.to(_device)

        _transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet channel stats
                std=[0.229, 0.224, 0.225],
            ),
        ])
        _model = model
        return _model


def bytes_to_image(image_bytes: bytes) -> Image.Image:
    """Decode raw uploaded bytes into a PIL RGB image."""
    image = Image.open(io.BytesIO(image_bytes))
    return image.convert("RGB")


@torch.no_grad()
def encode_image(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL image into a normalized L2 embedding vector (float32, shape (2048,)).

    Normalizing to unit length lets us use a simple inner-product FAISS index
    for cosine similarity search instead of maintaining a separate metric.
    """
    model = _load_model()
    tensor = _transform(image).unsqueeze(0).to(_device)  # (1, 3, 224, 224)

    features = model(tensor)                 # (1, 2048, 1, 1)
    features = features.squeeze().cpu().numpy().astype("float32")  # (2048,)

    norm = np.linalg.norm(features)
    if norm > 0:
        features = features / norm

    return features


@torch.no_grad()
def encode_images_batch(images: list[Image.Image]) -> np.ndarray:
    """Batched version of encode_image, useful when bulk-indexing a catalog."""
    model = _load_model()
    tensors = torch.stack([_transform(img) for img in images]).to(_device)

    features = model(tensors)                            # (B, 2048, 1, 1)
    features = features.squeeze(-1).squeeze(-1).cpu().numpy().astype("float32")

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    features = features / norms

    return features
