"""
Domain-Gap Evaluation: Run our trained EfficientNetB0 on Tomato-Village and
PlantDoc datasets WITHOUT retraining. Report per-class and aggregate metrics
with restricted PlantVillage baselines for fair comparison.

Usage:  python models/domain_gap_eval.py
Output: models/domain_gap_evaluation.json
"""
import json, os, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT = Path(r"G:\BTECH\Major Project\Project\An-Explainable-Digital-Twin-for-Data-Driven-Irrigation-Management-in-Tomato-Plant-Cultivation")
MODEL_PATH  = PROJECT / "models" / "efficientnet_plant_disease.pth"
META_PATH   = PROJECT / "models" / "image_model_meta.json"
CM_PATH     = PROJECT / "models" / "confusion_matrix.json"
OUTPUT_PATH = PROJECT / "models" / "domain_gap_evaluation.json"

TV_TEST  = PROJECT / "data" / "datasets" / "TomatoVillage_repo" / "Variant-a(Multiclass Classification)" / "test"
PD_TEST  = PROJECT / "data" / "datasets" / "PlantDoc_repo" / "test"
PD_TRAIN = PROJECT / "data" / "datasets" / "PlantDoc_repo" / "train"

# ── Class mappings (approved in implementation plan) ──────────────────────
TOMATO_VILLAGE_MAP = {
    "Early_blight": "Tomato_Early_blight",
    "Late_blight":  "Tomato_Late_blight",
    "Healthy":      "Tomato_healthy",
}

PLANTDOC_MAP = {
    "Tomato Early blight leaf":    "Tomato_Early_blight",
    "Tomato leaf late blight":     "Tomato_Late_blight",
    "Tomato leaf bacterial spot":  "Tomato_Bacterial_spot",
    "Tomato Septoria leaf spot":   "Tomato_Septoria_leaf_spot",
    "Tomato leaf mosaic virus":    "Tomato__Tomato_mosaic_virus",
    "Tomato leaf yellow virus":    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato mold leaf":            "Tomato_Leaf_Mold",
    "Tomato leaf":                 "Tomato_healthy",
}

# ── Load model ────────────────────────────────────────────────────────────
with open(META_PATH) as f:
    meta = json.load(f)
CLASS_NAMES = meta["class_names"]   # ordered list of 10 PlantVillage classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
ckpt = torch.load(MODEL_PATH, map_location=device)
# Checkpoint is a dict with 'model_state' key, not raw state_dict
model.load_state_dict(ckpt["model_state"])
model.to(device)
model.eval()

# Same transforms as training/evaluation (match image_inference.py)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Inference helpers ─────────────────────────────────────────────────────
@torch.no_grad()
def predict_image(img_path):
    """Return (predicted_class_name, confidence, top2_classes)."""
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        return None, None, None   # skip corrupt images
    tensor = transform(img).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    top2 = probs.topk(2)
    pred_idx = top2.indices[0].item()
    conf = top2.values[0].item()
    return CLASS_NAMES[pred_idx], conf, [(CLASS_NAMES[top2.indices[i].item()], top2.values[i].item()) for i in range(2)]


def evaluate_dataset(data_root, class_map, dataset_name):
    """
    Evaluate model on a dataset.
    data_root: Path to the root containing class folders.
    class_map: dict mapping folder_name → our PlantVillage class name.
    Returns evaluation dict.
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING: {dataset_name}")
    print(f"{'='*70}")

    # Collect images
    images = []   # list of (img_path, ground_truth_our_class)
    excluded = defaultdict(int)
    
    if not data_root.exists():
        print(f"  ERROR: {data_root} does not exist!")
        return None

    for folder in sorted(data_root.iterdir()):
        if not folder.is_dir():
            continue
        folder_name = folder.name
        if folder_name in class_map:
            our_class = class_map[folder_name]
            img_files = [f for f in folder.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")]
            for img_f in img_files:
                images.append((img_f, our_class))
        else:
            n_files = len([f for f in folder.iterdir() if f.is_file()])
            excluded[folder_name] = n_files

    print(f"  Mapped classes   : {len(class_map)}")
    print(f"  Excluded classes : {len(excluded)}")
    for exc_name, exc_count in sorted(excluded.items()):
        print(f"    - {exc_name}: {exc_count} images excluded")
    print(f"  Total eval images: {len(images)}")

    if len(images) == 0:
        print("  NO IMAGES FOUND — skipping.")
        return None

    # Run inference
    correct = 0
    total = 0
    skipped = 0
    per_class_tp = defaultdict(int)
    per_class_fp = defaultdict(int)
    per_class_fn = defaultdict(int)
    per_class_support = defaultdict(int)
    misclassifications = []   # for analysis

    t0 = time.perf_counter()
    for i, (img_path, gt_class) in enumerate(images):
        pred_class, conf, top2 = predict_image(img_path)
        if pred_class is None:
            skipped += 1
            continue
        
        total += 1
        per_class_support[gt_class] += 1
        
        if pred_class == gt_class:
            correct += 1
            per_class_tp[gt_class] += 1
        else:
            per_class_fn[gt_class] += 1
            per_class_fp[pred_class] += 1
            misclassifications.append({
                "file": str(img_path.name),
                "ground_truth": gt_class,
                "predicted": pred_class,
                "confidence": round(conf, 4),
            })
        
        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            print(f"    [{i+1}/{len(images)}]  acc so far: {correct/total:.4f}  ({elapsed:.1f}s)")

    elapsed = time.perf_counter() - t0
    print(f"\n  Inference time: {elapsed:.1f}s ({elapsed/max(total,1)*1000:.1f}ms/image)")
    if skipped:
        print(f"  Skipped {skipped} corrupt/unreadable images")

    # Compute metrics
    accuracy = correct / total if total > 0 else 0
    
    # Per-class P/R/F1
    our_matched_classes = sorted(set(class_map.values()))
    per_class_results = []
    f1_sum = 0
    for cls in our_matched_classes:
        tp = per_class_tp[cls]
        fp = per_class_fp[cls]
        fn = per_class_fn[cls]
        support = per_class_support[cls]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_sum += f1
        per_class_results.append({
            "class": cls,
            "tp": tp, "fp": fp, "fn": fn, "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })
        print(f"  {cls:50s}  {tp:3d}/{support:3d}  P={precision:.4f} R={recall:.4f} F1={f1:.4f}")
    
    macro_f1 = f1_sum / len(our_matched_classes) if our_matched_classes else 0

    print(f"\n  Overall accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Macro F1         : {macro_f1:.4f}")
    print(f"  Total correct    : {correct}/{total}")

    result = {
        "dataset": dataset_name,
        "total_images": total,
        "skipped_corrupt": skipped,
        "excluded_classes": dict(excluded),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "correct": correct,
        "total": total,
        "per_class": per_class_results,
        "inference_time_s": round(elapsed, 1),
        "ms_per_image": round(elapsed / max(total, 1) * 1000, 1),
        "top_misclassifications": sorted(misclassifications, key=lambda x: -x["confidence"])[:20],
    }
    return result


# ── Compute restricted PlantVillage baselines ─────────────────────────────
def compute_restricted_baseline(class_subset_names):
    """From our confusion matrix, compute accuracy/macro-F1 restricted to given classes."""
    with open(CM_PATH) as f:
        cm = json.load(f)
    names = cm["class_names"]
    matrix = cm["matrix"]
    per_class = cm["per_class"]
    
    indices = [names.index(c) for c in class_subset_names]
    total_correct = sum(matrix[i][i] for i in indices)
    total_samples = sum(sum(matrix[i]) for i in indices)
    acc = total_correct / total_samples if total_samples > 0 else 0
    macro_f1 = sum(per_class[i]["f1"] for i in indices) / len(indices)
    
    per_cls = []
    for i in indices:
        per_cls.append({
            "class": names[i],
            "tp": per_class[i]["tp"],
            "support": per_class[i]["support"],
            "precision": per_class[i]["precision"],
            "recall": per_class[i]["recall"],
            "f1": per_class[i]["f1"],
        })
    
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "total_samples": total_samples,
        "per_class": per_cls,
        "threshold_80pct": round(acc * 0.80, 4),
        "threshold_60pct": round(acc * 0.60, 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading model...")
    # model is already loaded at module level

    # 1. Restricted baselines
    tv_matched = sorted(set(TOMATO_VILLAGE_MAP.values()))
    pd_matched = sorted(set(PLANTDOC_MAP.values()))

    tv_baseline = compute_restricted_baseline(tv_matched)
    pd_baseline = compute_restricted_baseline(pd_matched)

    print(f"\nTV restricted baseline: {tv_baseline['accuracy']:.4f} ({tv_baseline['accuracy']*100:.2f}%)")
    print(f"PD restricted baseline: {pd_baseline['accuracy']:.4f} ({pd_baseline['accuracy']*100:.2f}%)")

    # 2. Evaluate Tomato-Village
    tv_result = evaluate_dataset(TV_TEST, TOMATO_VILLAGE_MAP, "Tomato-Village (Gehlot 2023)")

    # 3. Evaluate PlantDoc — use BOTH test and train as external validation
    # (we never trained on PlantDoc, so both splits are novel to our model)
    pd_test_result = evaluate_dataset(PD_TEST, PLANTDOC_MAP, "PlantDoc-test (Singh 2019)")
    pd_train_result = evaluate_dataset(PD_TRAIN, PLANTDOC_MAP, "PlantDoc-train (Singh 2019)")

    # Combine PlantDoc test+train for aggregate number
    pd_combined = None
    if pd_test_result and pd_train_result:
        combined_correct = pd_test_result["correct"] + pd_train_result["correct"]
        combined_total = pd_test_result["total"] + pd_train_result["total"]
        combined_acc = combined_correct / combined_total if combined_total > 0 else 0
        # Combine per-class
        combined_per_class = []
        for test_cls in pd_test_result["per_class"]:
            train_cls = next((t for t in pd_train_result["per_class"] if t["class"] == test_cls["class"]), None)
            if train_cls:
                tp = test_cls["tp"] + train_cls["tp"]
                fp = test_cls["fp"] + train_cls["fp"]
                fn = test_cls["fn"] + train_cls["fn"]
                support = test_cls["support"] + train_cls["support"]
                p = tp / (tp + fp) if (tp + fp) > 0 else 0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
                combined_per_class.append({
                    "class": test_cls["class"], "tp": tp, "fp": fp, "fn": fn,
                    "support": support, "precision": round(p, 4),
                    "recall": round(r, 4), "f1": round(f1, 4),
                })
        combined_macro_f1 = sum(c["f1"] for c in combined_per_class) / len(combined_per_class) if combined_per_class else 0
        pd_combined = {
            "dataset": "PlantDoc-combined (test+train)",
            "accuracy": round(combined_acc, 4),
            "macro_f1": round(combined_macro_f1, 4),
            "correct": combined_correct,
            "total": combined_total,
            "per_class": combined_per_class,
        }

    # 4. Tier interpretation
    def interpret_tier(result_acc, baseline_acc, dataset_label):
        if result_acc is None:
            return "SKIPPED — dataset not available"
        retention = result_acc / baseline_acc if baseline_acc > 0 else 0
        pct = retention * 100
        if retention >= 0.80:
            tier = "STRENGTH"
            desc = f">=80% retained ({pct:.1f}%) -> document as strength"
        elif retention >= 0.60:
            tier = "LIMITATION"
            desc = f"60-80% retained ({pct:.1f}%) -> document as honest limitation"
        else:
            tier = "SIGNIFICANT_DROP"
            desc = f"<60% retained ({pct:.1f}%) -> consider fine-tune plan"
        print(f"\n  {dataset_label}: {result_acc:.4f} vs baseline {baseline_acc:.4f} -> {pct:.1f}% retained -> {tier}")
        print(f"    {desc}")
        return {"retention_pct": round(pct, 1), "tier": tier, "description": desc}

    print("\n" + "=" * 70)
    print("TIER INTERPRETATION (against restricted baselines)")
    print("=" * 70)

    tv_tier = interpret_tier(
        tv_result["accuracy"] if tv_result else None,
        tv_baseline["accuracy"],
        "Tomato-Village"
    )

    pd_tier = interpret_tier(
        pd_combined["accuracy"] if pd_combined else (pd_test_result["accuracy"] if pd_test_result else None),
        pd_baseline["accuracy"],
        "PlantDoc-combined"
    )

    # 5. Save results
    output = {
        "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "EfficientNetB0 (PlantVillage, 10 classes, 93.88% test acc)",
        "restricted_baselines": {
            "tomato_village_3class": tv_baseline,
            "plantdoc_8class": pd_baseline,
        },
        "results": {
            "tomato_village": tv_result,
            "plantdoc_test": pd_test_result,
            "plantdoc_train": pd_train_result,
            "plantdoc_combined": pd_combined,
        },
        "tier_interpretation": {
            "tomato_village": tv_tier,
            "plantdoc": pd_tier,
        },
        "class_mappings": {
            "tomato_village": TOMATO_VILLAGE_MAP,
            "plantdoc": PLANTDOC_MAP,
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_PATH}")
