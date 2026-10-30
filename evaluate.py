
import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_large


def load_model(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint bulunamadı: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    architecture = checkpoint.get(
        "architecture", "mobilenet_v3_large"
    )

    if architecture != "mobilenet_v3_large":
        raise ValueError("Desteklenmeyen model mimarisi")

    class_to_idx = checkpoint["class_to_idx"]
    num_classes = len(class_to_idx)

    if sorted(class_to_idx.values()) != list(range(num_classes)):
        raise ValueError("Geçersiz class_to_idx eşlemesi")

    model = mobilenet_v3_large(weights=None)

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        num_classes
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    use_fp16 = (
        checkpoint.get("precision") == "fp16"
        and device.type == "cuda"
    )

    model = model.half() if use_fp16 else model.float()
    model = model.to(device).eval()

    return model, class_to_idx, use_fp16


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])


def predict_logits(model, images, use_tta=False):
    original_logits = model(images)

    if not use_tta:
        return original_logits

    flipped_images = torch.flip(images, dims=[3])
    flipped_logits = model(flipped_images)

    return (original_logits + flipped_logits) / 2


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure_speed(
    model,
    dataset,
    device,
    use_fp16,
    use_tta=False,
    repetitions=3,
    samples_per_run=100
):
    count = min(samples_per_run, len(dataset))

    if count == 0:
        raise ValueError("Hız ölçümü için veri bulunamadı")

    indices = torch.linspace(
        0, len(dataset) - 1, steps=count
    ).long().tolist()

    images = []

    for idx in indices:
        image, _ = dataset[idx]
        image = image.unsqueeze(0).to(device)

        if use_fp16:
            image = image.half()

        images.append(image)

    run_means = []
    all_times = []

    with torch.inference_mode():

        # Warm-up
        for i in range(30):
            predict_logits(
                model,
                images[i % count],
                use_tta=use_tta
            )

        synchronize(device)

        # Üç ayrı ölçüm turu
        for run in range(repetitions):
            times = []

            for image in images:
                synchronize(device)

                start = time.perf_counter()

                predict_logits(
                    model,
                    image,
                    use_tta=use_tta
                )

                synchronize(device)

                elapsed_ms = (
                    time.perf_counter() - start
                ) * 1000

                times.append(elapsed_ms)

            run_means.append(statistics.mean(times))
            all_times.extend(times)

    sorted_times = sorted(all_times)

    p95_index = max(
        0,
        math.ceil(0.95 * len(sorted_times)) - 1
    )

    return {
        "mean_ms": statistics.mean(all_times),
        "median_ms": statistics.median(all_times),
        "std_ms": (
            statistics.stdev(all_times)
            if len(all_times) > 1 else 0.0
        ),
        "p95_ms": sorted_times[p95_index],
        "run_means_ms": run_means,
        "median_run_mean_ms": statistics.median(run_means),
        "num_measurements": len(all_times),
        "num_runs": repetitions,
        "batch_size": 1,
        "scope": "model_forward_only"
    }


def calculate_class_metrics(y_true, y_pred, class_to_idx):
    idx_to_class = {
        idx: name
        for name, idx in class_to_idx.items()
    }

    num_classes = len(class_to_idx)

    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    pair_counts = Counter(zip(y_true, y_pred))

    per_class = {}
    precisions = []
    recalls = []
    f1_scores = []

    for idx in range(num_classes):
        tp = pair_counts[(idx, idx)]
        fp = pred_counts[idx] - tp
        fn = true_counts[idx] - tp

        precision = (
            tp / (tp + fp) if tp + fp else 0.0
        )

        recall = (
            tp / (tp + fn) if tp + fn else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        name = idx_to_class[idx]

        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": true_counts[idx]
        }

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    confusions = Counter(
        (idx_to_class[t], idx_to_class[p])
        for t, p in zip(y_true, y_pred)
        if t != p
    )

    top_confusions = [
        {
            "true_class": true_name,
            "predicted_class": pred_name,
            "count": count
        }
        for (true_name, pred_name), count
        in confusions.most_common(10)
    ]

    return {
        "macro_precision": statistics.mean(precisions),
        "macro_recall": statistics.mean(recalls),
        "macro_f1": statistics.mean(f1_scores),
        "per_class": per_class,
        "top_confusions": top_confusions
    }


def evaluate(
    checkpoint_path,
    data_dir,
    output_path=None,
    batch_size=32,
    use_tta=False
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, class_to_idx, use_fp16 = load_model(
        checkpoint_path, device
    )

    data_dir = Path(data_dir)

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Veri klasörü bulunamadı: {data_dir}"
        )

    dataset = datasets.ImageFolder(
        str(data_dir),
        transform=get_transform()
    )

    original_classes = dataset.classes

    unknown_classes = (
        set(original_classes) - set(class_to_idx)
    )

    if unknown_classes:
        raise ValueError(
            "Checkpoint'te bulunmayan sınıflar: "
            f"{sorted(unknown_classes)}"
        )

    # ImageFolder etiketlerini eğitim indekslerine eşleştir
    dataset.samples = [
        (path, class_to_idx[original_classes[label]])
        for path, label in dataset.samples
    ]

    dataset.imgs = dataset.samples

    dataset.targets = [
        label for _, label in dataset.samples
    ]

    dataset.class_to_idx = {
        name: class_to_idx[name]
        for name in original_classes
    }

    dataset.classes = [
        name for name, idx in sorted(
            class_to_idx.items(),
            key=lambda item: item[1]
        )
    ]

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda")
    )

    y_true = []
    y_pred = []

    # 1. Accuracy ve sınıf bazlı metrikler
    with torch.inference_mode():
        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            if use_fp16:
                images = images.half()

            outputs = predict_logits(
                model,
                images,
                use_tta=use_tta
            )

            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.cpu().tolist())

    total = len(y_true)

    correct = sum(
        t == p
        for t, p in zip(y_true, y_pred)
    )

    accuracy = 100 * correct / total

    class_metrics = calculate_class_metrics(
        y_true,
        y_pred,
        class_to_idx
    )

    # 2. Tek görüntü inference latency
    speed = measure_speed(
        model,
        dataset,
        device,
        use_fp16,
        use_tta=use_tta
    )

    # 3. Checkpoint boyutu
    size_mib = (
        Path(checkpoint_path).stat().st_size
        / 1024**2
    )

    results = {
        "model": "MobileNetV3-Large",
        "checkpoint": str(checkpoint_path),
        "precision": "fp16" if use_fp16 else "fp32",
        "device": str(device),
        "tta": use_tta,
        "accuracy_percent": accuracy,
        "correct": correct,
        "total": total,
        "model_size_mib": size_mib,
        "macro_precision": class_metrics["macro_precision"],
        "macro_recall": class_metrics["macro_recall"],
        "macro_f1": class_metrics["macro_f1"],
        "per_class": class_metrics["per_class"],
        "top_confusions": class_metrics["top_confusions"],
        "inference": speed
    }

    print("\n=== MODEL EVALUATION V3 ===")
    print("Checkpoint:", checkpoint_path)
    print("Precision:", results["precision"].upper())
    print("TTA:", "Enabled" if use_tta else "Disabled")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct}/{total}")
    print(f"Macro Precision: {results['macro_precision']:.4f}")
    print(f"Macro Recall: {results['macro_recall']:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Model size: {size_mib:.2f} MiB")
    print(f"Mean inference: {speed['mean_ms']:.3f} ms")
    print(f"Median inference: {speed['median_ms']:.3f} ms")
    print(f"Std inference: {speed['std_ms']:.3f} ms")
    print(f"P95 inference: {speed['p95_ms']:.3f} ms")
    print(
        f"Median run mean: "
        f"{speed['median_run_mean_ms']:.3f} ms"
    )
    print("Device:", device)

    print("\nTop 5 confusions:")

    for item in results["top_confusions"][:5]:
        print(
            f"  {item['true_class']} -> "
            f"{item['predicted_class']}: {item['count']}"
        )

    if output_path:
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                results,
                f,
                indent=4,
                ensure_ascii=False
            )

        print(f"\nResults saved: {output_path}")

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    parser.add_argument(
        "--data-dir",
        required=True
    )

    parser.add_argument(
        "--output",
        default=None
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable horizontal flip TTA"
    )

    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        use_tta=args.tta
    )
