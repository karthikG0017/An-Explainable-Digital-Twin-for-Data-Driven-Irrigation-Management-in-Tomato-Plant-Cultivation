"""
image_inference.py
------------------
Loads the EfficientNetB0 plant-disease model once and exposes a single
function: predict(pil_image) -> dict[class_name, probability]

Kept in a separate module so the Flask app can lazy-load it only on the
first image request, keeping cold-start time fast for sensor-only requests.
"""

import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ── Paths (relative to project root, resolved at import time) ─────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.dirname(_HERE)
_MODEL_PATH  = os.path.join(_ROOT, "models", "efficientnet_plant_disease.pth")
_META_PATH   = os.path.join(_ROOT, "models", "image_model_meta.json")

IMG_SIZE = 224
_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Module-level singletons (loaded once) ─────────────────────────────────────
_model       = None
_class_names = None
_device      = None


def _load():
    global _model, _class_names, _device
    if _model is not None:
        return  # already loaded

    with open(_META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    _class_names = meta["class_names"]
    num_classes  = len(_class_names)

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt  = torch.load(_MODEL_PATH, map_location=_device, weights_only=False)
    m     = models.efficientnet_b0(weights=None)
    in_f  = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.3, inplace=True),
        nn.Linear(in_f, num_classes),
    )
    m.load_state_dict(ckpt["model_state"])
    m.eval().to(_device)
    _model = m


def predict(pil_image: Image.Image) -> dict:
    """
    Run inference on a single PIL image.

    Returns
    -------
    dict  {class_name: probability}  — all 10 classes, probabilities sum to 1.
          Sorted descending by probability.
    """
    _load()

    # Ensure RGB (handles PNG with alpha channel, grayscale uploads, etc.)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    tensor = _TRANSFORM(pil_image).unsqueeze(0).to(_device)
    with torch.no_grad():
        probs = torch.softmax(_model(tensor), dim=1).cpu().squeeze().tolist()

    result = dict(zip(_class_names, probs))
    # Sort descending so the top class is first (cosmetic, dict order preserved)
    return dict(sorted(result.items(), key=lambda x: -x[1]))


def get_class_names() -> list:
    """Return the ordered list of class names (loads model metadata)."""
    _load()
    return list(_class_names)
