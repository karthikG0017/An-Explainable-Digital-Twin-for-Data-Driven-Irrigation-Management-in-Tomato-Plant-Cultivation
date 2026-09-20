"""
lime_image_model.py
-------------------
Item 4b: EfficientNetB0 fine-tuning on PlantVillage tomato images,
         followed by LIME explainability on the trained model.

TWO-PHASE SCRIPT
================
Phase 1 — Training (run once when PlantVillage images are available):
  - Loads EfficientNetB0 with ImageNet pretrained weights
  - Replaces the classifier head for N tomato disease classes
  - Fine-tunes for a configurable number of epochs
  - Saves the model to models/efficientnet_plant_disease.pth

Phase 2 — LIME explanation (run after training):
  - Loads the saved model
  - Picks one test image per class (or uses any supplied image)
  - Runs LIME ImageExplainer to identify which image regions drove the
    disease classification
  - Saves LIME overlay images to models/lime_outputs/
  - Prints a per-class summary of the most influential image region

IMPORTANT NOTES ON METHODOLOGY
===============================
  - LIME perturbs the image by masking random superpixels and observing
    how the model's output probability changes. The result shows WHICH
    regions the model attended to, NOT why the plant is diseased.
  - LIME explanations are LOCAL (per-image) approximations, not global
    feature rankings. Do not average LIME results across images and call
    it a global importance score.
  - Domain-gap validation datasets (PlantDoc, Tomato-Village) are
    NEVER used for training or fine-tuning, only for held-out testing.

DATA STRUCTURE EXPECTED
=======================
  data/datasets/PlantVillage/
      Tomato_Bacterial_spot/       (images: *.jpg or *.JPG)
      Tomato_Early_blight/
      Tomato_healthy/
      ... (other tomato disease folders)

If PlantVillage images are not yet available, the script reports
a clear message and exits cleanly — it does not crash.

Input   : data/datasets/PlantVillage/   (ImageFolder structure)
Output  : models/efficientnet_plant_disease.pth
          models/lime_outputs/<class_name>_lime.png  (one per class)
          printed per-image LIME summary
"""

import os
import sys
import json
import pickle
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, transforms, models
from lime import lime_image
from skimage.segmentation import mark_boundaries

# ── Config ────────────────────────────────────────────────────────────────────
ROOT_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR   = os.path.dirname(ROOT_DIR)
DATA_DIR      = os.path.join(PROJECT_DIR, "data", "datasets", "PlantVillage")
MODEL_OUT     = os.path.join(ROOT_DIR, "efficientnet_plant_disease.pth")
LIME_OUT_DIR  = os.path.join(ROOT_DIR, "lime_outputs")
META_OUT      = os.path.join(ROOT_DIR, "image_model_meta.json")

# Training hyperparameters
BATCH_SIZE   = 32
EPOCHS       = 10          # increase to 20+ for full training
LR           = 1e-4        # low LR: pretrained backbone needs gentle fine-tuning
IMG_SIZE     = 224         # EfficientNetB0 native input size
TRAIN_FRAC   = 0.80
VAL_FRAC     = 0.10        # remaining 10% = test
SEED         = 42

# LIME config
LIME_NUM_SAMPLES   = 1000   # perturbation samples per image (higher = slower but more stable)
LIME_NUM_FEATURES  = 10     # number of superpixel regions to highlight
LIME_TOP_LABELS    = 1      # explain the top predicted class

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Transforms ────────────────────────────────────────────────────────────────
def get_transforms():
    """
    Training transform includes augmentation (random flip, rotation, color jitter)
    to reduce overfitting on the relatively small PlantVillage set.
    Validation/test transform uses only resize + center crop + normalize.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    return train_tf, val_tf


# ── Model definition ──────────────────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    """
    EfficientNetB0 with pretrained ImageNet weights.
    Only the classifier head is replaced; the backbone starts frozen
    then is unfrozen after a warmup phase (transfer learning strategy).
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Freeze all backbone parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final classifier for our number of classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    # Classifier head is trainable from the start
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model.to(DEVICE)


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all backbone layers after warmup epochs."""
    for param in model.parameters():
        param.requires_grad = True


# ── Phase 1: Training ─────────────────────────────────────────────────────────
def train_model(dataset_root: str) -> tuple[nn.Module, list[str]]:
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_tf, val_tf = get_transforms()

    # Load full dataset with training transforms first (for split)
    full_dataset = datasets.ImageFolder(dataset_root, transform=train_tf)
    class_names  = full_dataset.classes
    num_classes  = len(class_names)

    print(f"  Classes ({num_classes}): {class_names}")
    print(f"  Total images: {len(full_dataset):,}")

    # Time-independent split (random, by image — no temporal ordering for images)
    n_total = len(full_dataset)
    n_train = int(n_total * TRAIN_FRAC)
    n_val   = int(n_total * VAL_FRAC)
    n_test  = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )
    # Apply val transform to val/test subsets
    val_ds.dataset  = datasets.ImageFolder(dataset_root, transform=val_tf)
    test_ds.dataset = datasets.ImageFolder(dataset_root, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = build_model(num_classes)
    criterion = nn.CrossEntropyLoss()

    # Phase 1: train classifier head only (frozen backbone) — 3 warmup epochs
    WARMUP_EPOCHS = min(3, EPOCHS)
    optimizer_warmup = torch.optim.Adam(model.classifier.parameters(), lr=LR)

    print(f"\n  Warmup ({WARMUP_EPOCHS} epochs, backbone frozen) ...")
    for epoch in range(WARMUP_EPOCHS):
        _run_epoch(model, train_loader, criterion, optimizer_warmup, train=True)
        val_acc = _run_epoch(model, val_loader, criterion, None, train=False)
        print(f"    Epoch {epoch+1}/{WARMUP_EPOCHS}  val_acc={val_acc:.4f}")

    # Phase 2: unfreeze and fine-tune the full network with lower LR
    if EPOCHS > WARMUP_EPOCHS:
        unfreeze_backbone(model)
        optimizer_full = torch.optim.Adam(model.parameters(), lr=LR / 10)
        scheduler      = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_full, T_max=EPOCHS - WARMUP_EPOCHS
        )
        print(f"\n  Full fine-tune ({EPOCHS - WARMUP_EPOCHS} epochs, backbone unfrozen) ...")
        for epoch in range(EPOCHS - WARMUP_EPOCHS):
            _run_epoch(model, train_loader, criterion, optimizer_full, train=True)
            val_acc = _run_epoch(model, val_loader, criterion, None, train=False)
            scheduler.step()
            print(f"    Epoch {WARMUP_EPOCHS+epoch+1}/{EPOCHS}  val_acc={val_acc:.4f}")

    # Save
    torch.save({
        "model_state": model.state_dict(),
        "class_names": class_names,
        "num_classes": num_classes,
        "img_size":    IMG_SIZE,
    }, MODEL_OUT)

    meta = {"class_names": class_names, "num_classes": num_classes,
            "n_train": n_train, "n_val": n_val, "n_test": n_test}
    with open(META_OUT, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Model saved -> {MODEL_OUT}")
    return model, class_names


def _run_epoch(model, loader, criterion, optimizer, train: bool) -> float:
    model.train(train)
    correct = total = 0
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            preds    = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total if total > 0 else 0.0


# ── Phase 2: LIME explanation ─────────────────────────────────────────────────
def explain_with_lime(model: nn.Module, class_names: list[str],
                      dataset_root: str) -> None:
    """
    For each class, finds one test image and produces a LIME explanation.
    LIME masks random superpixels, queries the model, and identifies
    which regions most influenced the predicted class probability.
    """
    os.makedirs(LIME_OUT_DIR, exist_ok=True)
    _, val_tf = get_transforms()

    # Build a predict_fn that accepts a batch of numpy uint8 images [N, H, W, 3]
    # and returns class probabilities [N, num_classes].
    # LIME works in image space (numpy, 0-255 uint8); the model needs normalized tensors.
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
    imagenet_std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

    def predict_fn(images: np.ndarray) -> np.ndarray:
        """
        images: numpy array [N, H, W, 3], uint8 (0-255) — LIME's format.
        Returns: numpy array [N, num_classes] of softmax probabilities.
        """
        model.eval()
        batch = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
        batch = (batch.to(DEVICE) - imagenet_mean) / imagenet_std
        with torch.no_grad():
            logits = model(batch)
        return torch.softmax(logits, dim=1).cpu().numpy()

    explainer = lime_image.LimeImageExplainer(random_state=SEED)

    print(f"\n{'=' * 62}")
    print("  LIME Explanations — one image per class")
    print(f"{'=' * 62}")
    print("  NOTE: LIME is a LOCAL explainer. Each result describes which")
    print("  image regions drove THIS specific prediction. It is not a")
    print("  global measure of what the model learned about each disease.")
    print()

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_root, class_name)
        img_files = [f for f in os.listdir(class_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not img_files:
            print(f"  [{class_name}] No images found — skipping.")
            continue

        # Pick a deterministic test image (last 10% of sorted file list)
        img_files_sorted = sorted(img_files)
        test_pool = img_files_sorted[int(len(img_files_sorted) * 0.9):]
        img_path  = os.path.join(class_dir, test_pool[0])

        # Load as numpy uint8 (LIME's required format)
        img_pil  = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img_np   = np.array(img_pil)   # [H, W, 3], uint8

        # Get model prediction first
        prob = predict_fn(img_np[np.newaxis])[0]
        pred_class = int(prob.argmax())
        pred_conf  = float(prob[pred_class])
        correct    = (pred_class == class_idx)

        # Run LIME
        explanation = explainer.explain_instance(
            img_np,
            predict_fn,
            top_labels      = LIME_TOP_LABELS,
            hide_color      = 0,          # mask occulted regions with black
            num_samples     = LIME_NUM_SAMPLES,
            batch_size      = 32,
        )

        # Get the image with positive superpixels highlighted (pro-prediction regions)
        temp_img, mask = explanation.get_image_and_mask(
            pred_class,
            positive_only   = True,
            num_features    = LIME_NUM_FEATURES,
            hide_rest       = False,
        )

        # Overlay superpixel boundaries on the original image
        lime_overlay = (mark_boundaries(temp_img / 255.0, mask) * 255).astype(np.uint8)
        out_path = os.path.join(LIME_OUT_DIR, f"{class_name}_lime.png")
        Image.fromarray(lime_overlay).save(out_path)

        # Compute % of image area covered by top LIME regions
        area_pct = mask.sum() / mask.size * 100

        print(f"  [{class_name}]")
        print(f"    Image       : {os.path.basename(img_path)}")
        print(f"    Predicted   : {class_names[pred_class]}  (conf={pred_conf:.3f})")
        print(f"    Correct     : {'YES' if correct else 'NO'}")
        print(f"    LIME regions: {mask.sum():,} px  ({area_pct:.1f}% of image)")
        print(f"    Saved       : {out_path}")
        print()

    print(f"{'=' * 62}")
    print("  All LIME outputs saved to models/lime_outputs/")
    print("  Each PNG shows highlighted superpixels that pushed the model")
    print("  toward its prediction for that specific image.")
    print(f"{'=' * 62}")


# ── Load existing model ───────────────────────────────────────────────────────
def load_model() -> tuple[nn.Module, list[str]]:
    if not os.path.exists(MODEL_OUT):
        return None, None
    ckpt        = torch.load(MODEL_OUT, map_location=DEVICE, weights_only=False)
    class_names = ckpt["class_names"]
    model       = build_model(len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, class_names


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 62)
    print("  Item 4b — EfficientNetB0 + LIME (Image Disease Model)")
    print("=" * 62)
    print(f"  Device: {DEVICE}")

    # Check dataset
    if not os.path.isdir(DATA_DIR):
        print(f"\n  PlantVillage dataset not found at:")
        print(f"  {DATA_DIR}")
        print()
        print("  Expected folder structure:")
        print("    data/datasets/PlantVillage/")
        print("      Tomato_Bacterial_spot/   (*.jpg images)")
        print("      Tomato_Early_blight/")
        print("      Tomato_healthy/")
        print("      ... (other tomato subfolders)")
        print()
        print("  Download from: https://www.kaggle.com/datasets/emmarex/plantdisease")
        print("  Filter to tomato folders only before placing here.")
        print()
        print("  If you want to run LIME on an already-trained model,")
        print("  ensure models/efficientnet_plant_disease.pth also exists.")
        sys.exit(0)

    class_dirs = [d for d in os.listdir(DATA_DIR)
                  if os.path.isdir(os.path.join(DATA_DIR, d))]
    if not class_dirs:
        print("  PlantVillage folder is empty. Add tomato class subfolders.")
        sys.exit(0)

    print(f"\n  Found {len(class_dirs)} class folders in {DATA_DIR}")

    # Phase 1: train if model doesn't exist yet
    if os.path.exists(MODEL_OUT):
        print(f"\n  Existing model found: {MODEL_OUT}")
        print("  Skipping training — loading saved model ...")
        model, class_names = load_model()
        print(f"  Classes: {class_names}")
    else:
        print("\nPhase 1 — Training EfficientNetB0 ...")
        model, class_names = train_model(DATA_DIR)

    # Phase 2: LIME
    print("\nPhase 2 — Running LIME explanations ...")
    explain_with_lime(model, class_names, DATA_DIR)

    print("\nDone. Proceed to item 5 (fusion logic).")
