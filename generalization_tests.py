#!/usr/bin/env python3
"""Generalization & comparison tests for the CLIP flower classifier (3. Kişi deliverable).

Usage:
    python generalization_tests.py --task ood          # veri setinde olmayan türler
    python generalization_tests.py --task corruption   # görüntü bozulmalarına dayanıklılık
    python generalization_tests.py --task compare      # CLIP vs diğer modeller (results/*_metrics.json)
    python generalization_tests.py --task all          # hepsi
"""
import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt

from src.clip_utils import (
    build_class_prompts,
    encode_images,
    encode_text_prompts,
    list_class_images,
    load_cat_to_name,
    load_clip,
    open_image,
)
from src.corruptions import CORRUPTIONS, SEVERITIES
from src.ood_species import download_ood_samples

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
STYLE = {"surface": "#fcfcfb", "grid": "#e1e0d9", "ink": "#0b0b0b", "muted": "#898781", "axis": "#c3c2b7"}


def _style_axes(ax):
    ax.set_facecolor(STYLE["surface"])
    ax.grid(True, color=STYLE["grid"], linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(STYLE["axis"])
    ax.tick_params(colors=STYLE["muted"])
    ax.title.set_color(STYLE["ink"])
    ax.xaxis.label.set_color(STYLE["ink"])
    ax.yaxis.label.set_color(STYLE["ink"])


def setup():
    model, processor, device = load_clip()
    cat_to_name = load_cat_to_name()
    class_ids, prompts = build_class_prompts(cat_to_name)
    text_features = encode_text_prompts(model, processor, prompts, device)
    return model, processor, device, cat_to_name, class_ids, text_features


def predict(model, processor, device, text_features, images):
    feats = encode_images(model, processor, images, device)
    logits = feats @ text_features.T
    top_idx = logits.argmax(dim=-1)
    top_score = logits.gather(1, top_idx.unsqueeze(1)).squeeze(1)
    return top_idx.cpu().tolist(), top_score.cpu().tolist()


def run_ood(model, processor, device, cat_to_name, class_ids, text_features):
    ood_dir = Path("data/ood_samples")
    if not ood_dir.exists() or not any(p.is_dir() for p in ood_dir.iterdir()):
        print("OOD örnekleri indiriliyor (ilk çalıştırma)...")
        download_ood_samples(cat_to_name)

    rows = []
    for species_dir in sorted(p for p in ood_dir.iterdir() if p.is_dir()):
        species = species_dir.name.replace("_", " ")
        img_paths = sorted(species_dir.glob("*.jpg")) + sorted(species_dir.glob("*.png"))
        if not img_paths:
            continue
        images = [open_image(p) for p in img_paths]
        pred_idx, scores = predict(model, processor, device, text_features, images)
        for p, idx, score in zip(img_paths, pred_idx, scores):
            predicted_class = cat_to_name[class_ids[idx]]
            rows.append({
                "species": species,
                "image": str(p),
                "predicted_known_class": predicted_class,
                "confidence": round(score, 4),
            })
            print(f"[OOD] true={species:15s} -> predicted(102 sınıftan)={predicted_class:25s} conf={score:.3f}")

    avg_conf = sum(r["confidence"] for r in rows) / len(rows)

    # Karşılaştırma için: bilinen (in-distribution) test görüntülerinde ortalama top-1 confidence
    rng = random.Random(42)
    test_items = list_class_images("data/test")
    sample = rng.sample(test_items, min(100, len(test_items)))
    sample_images = [open_image(p) for p, _ in sample]
    _, id_scores = predict(model, processor, device, text_features, sample_images)
    avg_id_conf = sum(id_scores) / len(id_scores)

    out = {
        "num_images": len(rows),
        "avg_confidence_ood": round(avg_conf, 4),
        "avg_confidence_in_distribution": round(avg_id_conf, 4),
        "rows": rows,
    }
    with open(RESULTS_DIR / "ood_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nOrtalama confidence (OOD, 102 sınıf arasında zorla seçim): {avg_conf:.3f}")
    print(f"Ortalama confidence (in-distribution, bilinen türler):      {avg_id_conf:.3f}")
    return out


def run_corruption(model, processor, device, cat_to_name, class_ids, text_features, sample_size=200, seed=42):
    test_items = list_class_images("data/test")
    rng = random.Random(seed)
    sample = rng.sample(test_items, min(sample_size, len(test_items)))
    class_id_to_idx = {c: i for i, c in enumerate(class_ids)}

    def accuracy_for(corrupt_fn, severity):
        correct = 0
        batch_images, batch_true = [], []

        def flush():
            nonlocal correct, batch_images, batch_true
            if not batch_images:
                return
            pred_idx, _ = predict(model, processor, device, text_features, batch_images)
            correct += sum(p == t for p, t in zip(pred_idx, batch_true))
            batch_images, batch_true = [], []

        for path, true_class_id in sample:
            img = open_image(path)
            if corrupt_fn is not None:
                img = corrupt_fn(img, severity)
            batch_images.append(img)
            batch_true.append(class_id_to_idx[true_class_id])
            if len(batch_images) == 32:
                flush()
        flush()
        return correct / len(sample)

    clean_acc = accuracy_for(None, None)
    print(f"[corruption] clean accuracy (n={len(sample)}): {clean_acc:.4f}")

    results = {"clean_accuracy": round(clean_acc, 4), "sample_size": len(sample), "corruptions": {}}
    for name, fn in CORRUPTIONS.items():
        accs = []
        for severity in SEVERITIES:
            acc = accuracy_for(fn, severity)
            accs.append(round(acc, 4))
            print(f"[corruption] {name:18s} severity={severity} -> accuracy={acc:.4f}")
        results["corruptions"][name] = accs

    with open(RESULTS_DIR / "corruption_robustness.json", "w") as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=STYLE["surface"])
    ax.axhline(clean_acc, color=STYLE["muted"], linestyle="--", linewidth=1.5, label="Clean (bozulmamış)")
    for i, (name, accs) in enumerate(results["corruptions"].items()):
        ax.plot(SEVERITIES, accs, color=COLORS[i % len(COLORS)], linewidth=2, marker="o", markersize=5, label=name)
    ax.set_xlabel("Severity (1=hafif, 5=şiddetli)")
    ax.set_ylabel("Zero-shot accuracy")
    ax.set_title("CLIP: Görüntü Bozulmalarına Dayanıklılık")
    ax.set_xticks(SEVERITIES)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "corruption_robustness.png", dpi=150, facecolor=STYLE["surface"])
    plt.close(fig)
    print("Kaydedildi: results/corruption_robustness.json, results/corruption_robustness.png")
    return results


def run_compare():
    metrics_files = sorted(RESULTS_DIR.glob("*_metrics.json"))
    rows = []
    for f in metrics_files:
        with open(f) as fh:
            rows.append(json.load(fh))

    if not rows:
        print("Hiç *_metrics.json bulunamadı; önce clip.ipynb'yi çalıştırın.")
        return []

    header = f"{'model':30s} {'accuracy':>10s} {'ms/image':>10s} {'size (MB)':>10s}"
    print(header)
    for r in rows:
        print(f"{r.get('model_name', ''):30s} {r.get('accuracy', 0):10.4f} "
              f"{r.get('inference_ms_per_image', 0):10.2f} {r.get('model_size_mb', 0):10.2f}")

    others = [r for r in rows if "CLIP" not in r.get("model_name", "")]
    if not others:
        print("\nNot: Henüz diğer takım üyelerinin (EfficientNet-B0 / MobileNetV3 / EfficientNetV2-S) "
              "metrik dosyaları (results/<model>_metrics.json, aynı şema) eklenmedi. "
              "Eklendiklerinde bu script otomatik olarak karşılaştırmaya dahil edecek.")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor=STYLE["surface"])
    metric_defs = [
        ("accuracy", "Accuracy"),
        ("inference_ms_per_image", "Inference (ms/image)"),
        ("model_size_mb", "Model size (MB)"),
    ]
    names = [r.get("model_name", "?") for r in rows]
    for ax, (key, title) in zip(axes, metric_defs):
        values = [r.get(key, 0) for r in rows]
        ax.bar(range(len(rows)), values, color=[COLORS[i % len(COLORS)] for i in range(len(rows))])
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        _style_axes(ax)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparison_chart.png", dpi=150, facecolor=STYLE["surface"])
    plt.close(fig)

    fieldnames = ["model_name", "accuracy", "inference_ms_per_image", "model_size_mb", "device"]
    with open(RESULTS_DIR / "comparison_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print("Kaydedildi: results/comparison_table.csv, results/comparison_chart.png")
    return rows


def main():
    parser = argparse.ArgumentParser(description="CLIP genelleme/karşılaştırma testleri (3. Kişi).")
    parser.add_argument("--task", choices=["ood", "corruption", "compare", "all"], default="all")
    parser.add_argument("--corruption-sample-size", type=int, default=200)
    args = parser.parse_args()

    model = processor = device = cat_to_name = class_ids = text_features = None
    if args.task in ("ood", "corruption", "all"):
        model, processor, device, cat_to_name, class_ids, text_features = setup()

    if args.task in ("ood", "all"):
        run_ood(model, processor, device, cat_to_name, class_ids, text_features)
    if args.task in ("corruption", "all"):
        run_corruption(model, processor, device, cat_to_name, class_ids, text_features,
                        sample_size=args.corruption_sample_size)
    if args.task in ("compare", "all"):
        run_compare()


if __name__ == "__main__":
    main()
