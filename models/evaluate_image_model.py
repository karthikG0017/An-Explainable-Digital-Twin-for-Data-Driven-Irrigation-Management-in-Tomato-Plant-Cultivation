"""
evaluate_image_model.py
-----------------------
Post-training evaluation of the EfficientNetB0 plant disease classifier.

Produces:
  1. Full confusion matrix on the held-out test set (last 10% of images,
     same random_split seed as lime_image_model.py).
  2. Per-class precision, recall, F1, and support — surfacing which classes
     have weak recall or are confused with visually similar classes.
  3. LIME attention coverage analysis — reads the saved LIME PNG outputs,
     measures what % of each image was highlighted, and characterises
     whether the model shows tight lesion localisation or broad leaf attention.

IMPORTANT FRAMING NOTE:
  This script evaluates on the SAME PlantVillage test split used during
  training (intra-dataset test). Do NOT interpret these metrics as the
  model's performance on real field images — PlantVillage images are
  lab-controlled, high-quality, and spectrally cleaner than field photos.
  Domain-gap performance must be assessed separately using PlantDoc or
  Tomato-Village (which are held out and never used here).

Input  : models/efficientnet_plant_disease.pth
         models/image_model_meta.json
         data/datasets/PlantVillage/   (same folder as training)
         models/lime_outputs/*.png     (from lime_image_model.py)
Output : printed report + models/confusion_matrix.json (for Flask API use)
"""

import os
import json
import random

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(ROOT_DIR)
DATA_DIR     = os.path.join(PROJECT_DIR, "data", "datasets", "PlantVillage")
MODEL_PATH   = os.path.join(ROOT_DIR, "efficientnet_plant_disease.pth")
META_PATH    = os.path.join(ROOT_DIR, "image_model_meta.json")
LIME_DIR     = os.path.join(ROOT_DIR, "lime_outputs")
CM_OUT_PATH  = os.path.join(ROOT_DIR, "confusion_matrix.json")

IMG_SIZE   = 224
BATCH_SIZE = 64
SEED       = 42    # MUST match lime_image_model.py to get the same split
TRAIN_FRAC = 0.80
VAL_FRAC   = 0.10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Load model ─────────────────────────────────────────────────────────────────
def load_model(class_names):
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, len(class_names)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model.to(DEVICE)


# ── Reconstruct the SAME test split ───────────────────────────────────────────
def get_test_loader(class_names):
    """
    Reproduces the identical random_split from lime_image_model.py.
    Using the same SEED guarantees the test set is truly held-out.
    """
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    full_ds = datasets.ImageFolder(DATA_DIR, transform=val_tf)

    n_total = len(full_ds)
    n_train = int(n_total * TRAIN_FRAC)
    n_val   = int(n_total * VAL_FRAC)
    n_test  = n_total - n_train - n_val

    _, _, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )
    return DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ── Run inference ──────────────────────────────────────────────────────────────
def run_inference(model, loader, num_classes):
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return (np.array(all_labels), np.array(all_preds), np.array(all_probs))


# ── Confusion matrix (manual, no sklearn) ─────────────────────────────────────
def build_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


# ── Per-class metrics ──────────────────────────────────────────────────────────
def per_class_metrics(cm, class_names):
    metrics = []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        metrics.append({
            "class": name, "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "support": int(support),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
        })
    return metrics


# ── Print confusion matrix ─────────────────────────────────────────────────────
def print_confusion_matrix(cm, class_names):
    # Short labels for readability
    short = [n.replace("Tomato_", "").replace("Tomato__", "")[:14] for n in class_names]
    n = len(class_names)

    print(f"\n{'=' * 80}")
    print("  CONFUSION MATRIX (rows = actual, cols = predicted)")
    print(f"{'=' * 80}")
    print("  Abbrevations (column order):")
    for i, (s, full) in enumerate(zip(short, class_names)):
        print(f"    {i:2d}  {s:<18}  {full}")

    print()
    # Header row
    header = "  Actual \\ Pred  " + "".join(f"{i:>5}" for i in range(n))
    print(header)
    print("  " + "-" * (16 + 5 * n))

    for i in range(n):
        row_label = f"  {i:2d} {short[i]:<13}"
        row_vals  = ""
        for j in range(n):
            val = cm[i, j]
            # Highlight diagonal (correct) and off-diagonal errors
            if i == j:
                row_vals += f"{val:>5}"      # correct predictions
            elif val > 0:
                row_vals += f"{val:>5}"      # confusions
            else:
                row_vals += "    ."          # zero — cleaner to read
        print(row_label + row_vals)

    print()
    # Highlight top confusions (off-diagonal, non-zero)
    confusions = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], class_names[i], class_names[j]))
    confusions.sort(reverse=True)

    print("  Top confusions (actual -> predicted, count):")
    for count, actual, pred in confusions[:8]:
        a_short = actual.replace("Tomato_", "").replace("Tomato__", "")
        p_short = pred.replace("Tomato_", "").replace("Tomato__", "")
        print(f"    {count:>4}x  {a_short} -> {p_short}")
    print(f"{'=' * 80}")


# ── Print per-class metrics ────────────────────────────────────────────────────
def print_per_class_metrics(metrics, overall_acc):
    print(f"\n{'=' * 80}")
    print(f"  PER-CLASS METRICS  (overall test accuracy: {overall_acc*100:.2f}%)")
    print(f"{'=' * 80}")
    print(f"  {'Class':<44}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Support':>8}")
    print(f"  {'-'*44}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}")

    for m in sorted(metrics, key=lambda x: x["recall"]):
        short = m["class"].replace("Tomato__", "").replace("Tomato_", "")
        # Flag weak classes
        flag = ""
        if m["recall"] < 0.85:
            flag = " <-- LOW RECALL"
        if m["f1"] < 0.85:
            flag = flag or " <-- LOW F1"
        print(f"  {short:<44}  {m['precision']:>6.3f}  {m['recall']:>6.3f}  "
              f"{m['f1']:>6.3f}  {m['support']:>8}{flag}")

    print()
    avg_p = sum(m["precision"] for m in metrics) / len(metrics)
    avg_r = sum(m["recall"]    for m in metrics) / len(metrics)
    avg_f = sum(m["f1"]        for m in metrics) / len(metrics)
    print(f"  {'Macro average':<44}  {avg_p:>6.3f}  {avg_r:>6.3f}  {avg_f:>6.3f}")
    print()
    print("  INTERPRETATION NOTE:")
    print("  These are intra-dataset metrics (PlantVillage test split).")
    print("  PlantVillage images are lab-controlled; real-field performance")
    print("  will be lower. Domain-gap assessment requires PlantDoc/Tomato-Village.")
    print(f"{'=' * 80}")


# ── LIME coverage analysis ─────────────────────────────────────────────────────
def analyze_lime_coverage(class_names):
    """
    Reads each LIME output PNG, estimates the fraction of highlighted pixels
    by detecting superpixel boundary overlay colour (green tint from
    mark_boundaries), and characterises localisation quality.

    NOTE ON METHOD: mark_boundaries draws green outlines around selected
    superpixels. We detect the highlighted region by looking at pixels where
    the green channel is significantly boosted relative to red and blue —
    a simple but reliable heuristic for the mark_boundaries output format.
    This is an approximation; exact pixel counts would require storing the
    mask array directly (which the current lime_image_model.py does not do).
    A future iteration should save masks to disk for precise measurement.
    """
    print(f"\n{'=' * 80}")
    print("  LIME ATTENTION COVERAGE ANALYSIS")
    print(f"{'=' * 80}")
    print("  Measures: what fraction of each image did LIME highlight?")
    print("  Tight (<20%): lesion-localised. Broad (>35%): whole-leaf attention.")
    print()
    print(f"  {'Class':<44}  {'Highlighted%':>13}  {'Character'}")
    print(f"  {'-'*44}  {'-'*13}  {'-'*20}")

    coverage_vals = []
    results = []

    for class_name in class_names:
        png_path = os.path.join(LIME_DIR, f"{class_name}_lime.png")
        if not os.path.exists(png_path):
            print(f"  {class_name:<44}  (no PNG found)")
            continue

        img = np.array(Image.open(png_path).convert("RGB")).astype(float)
        H, W, _ = img.shape

        # mark_boundaries overlays green boundary lines on selected superpixels.
        # Highlighted pixels have elevated green relative to R and B channels.
        # Threshold: green channel > red+10 AND green > blue+10
        is_highlighted = (img[:,:,1] > img[:,:,0] + 10) & (img[:,:,1] > img[:,:,2] + 10)
        pct = is_highlighted.sum() / (H * W) * 100

        if pct < 15:
            char = "Tight (lesion-level)"
        elif pct < 30:
            char = "Moderate"
        else:
            char = "Broad (whole-leaf)"

        short = class_name.replace("Tomato__", "").replace("Tomato_", "")
        print(f"  {short:<44}  {pct:>12.1f}%  {char}")
        coverage_vals.append(pct)
        results.append({"class": class_name, "highlighted_pct": round(pct, 2),
                         "character": char})

    if coverage_vals:
        avg = sum(coverage_vals) / len(coverage_vals)
        print()
        print(f"  Average highlighted area across all classes: {avg:.1f}%")
        print()

        if avg < 15:
            verdict = "TIGHT — model attends primarily to lesion regions."
        elif avg < 30:
            verdict = "MODERATE — mix of lesion and surrounding leaf tissue."
        else:
            verdict = ("BROAD — model attends to large leaf regions, not just lesions.\n"
                       "  This is a known characteristic of CNN classifiers trained\n"
                       "  without explicit spatial supervision (e.g., no bounding-box\n"
                       "  or segmentation labels). The model may be using texture or\n"
                       "  colour statistics across the leaf rather than localised\n"
                       "  lesion morphology. This should be noted as a limitation\n"
                       "  in the thesis: LIME shows WHERE the model looked, but\n"
                       "  broad attention does not invalidate the classification —\n"
                       "  it means the model's discriminative signal is distributed\n"
                       "  across the image rather than concentrated at lesions.\n"
                       "  Explicit spatial supervision (weakly supervised localisation\n"
                       "  or GradCAM with bounding boxes) would be needed to force\n"
                       "  tighter lesion focus, which is out of scope for this project.")
        print(f"  Verdict: {verdict}")

    print(f"{'=' * 80}")
    return results


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 80)
    print("  Image Model Evaluation — Confusion Matrix + Per-class Metrics + LIME Coverage")
    print("=" * 80)
    print(f"  Device: {DEVICE}")

    # Load meta
    with open(META_PATH) as f:
        meta = json.load(f)
    class_names = meta["class_names"]
    num_classes = meta["num_classes"]
    print(f"  Classes: {num_classes}  |  Test set: {meta['n_test']} images")

    # Load model
    print("\nLoading model ...")
    model = load_model(class_names)

    # Get test data
    print("Reconstructing test split ...")
    test_loader = get_test_loader(class_names)

    # Inference
    print("Running inference on test set ...")
    y_true, y_pred, y_probs = run_inference(model, test_loader, num_classes)

    overall_acc = (y_true == y_pred).mean()

    # Confusion matrix
    cm = build_confusion_matrix(y_true, y_pred, num_classes)
    print_confusion_matrix(cm, class_names)

    # Per-class metrics
    metrics = per_class_metrics(cm, class_names)
    print_per_class_metrics(metrics, overall_acc)

    # LIME coverage
    lime_results = analyze_lime_coverage(class_names)

    # Save confusion matrix JSON for Flask API
    cm_data = {
        "class_names": class_names,
        "matrix":      cm.tolist(),
        "per_class":   metrics,
        "overall_acc": round(float(overall_acc), 4),
        "lime_coverage": lime_results,
    }
    with open(CM_OUT_PATH, "w") as f:
        json.dump(cm_data, f, indent=2)
    print(f"\n  Saved -> {CM_OUT_PATH}  (confusion matrix + metrics for Flask API)")
    print("\nDone.")
