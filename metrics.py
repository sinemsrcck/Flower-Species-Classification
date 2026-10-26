"""metrics.py — Değerlendirme yardımcıları (Kişi 4).

Accuracy, precision, recall, F1, confusion matrix, eğitim grafikleri,
hatalı tahmin analizi, CPU inference hızı, model boyutu ve yarışma puanı.
"""
import json
import os
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


@torch.no_grad()
def get_predictions(model, loader, device):
    """Bir DataLoader üzerinde (y_true, y_pred, y_prob) döndürür."""
    model.eval()
    ys, preds, probs = [], [], []
    for x, y in loader:
        out = model(x.to(device, non_blocking=True))
        p = out.softmax(1).cpu()
        probs.append(p)
        preds.append(p.argmax(1))
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(preds).numpy(), torch.cat(probs).numpy()


def compute_metrics(y_true, y_pred, y_prob=None, k=5):
    """Top-1 (ve varsa top-k) accuracy ile macro/weighted precision, recall, F1."""
    res = {"accuracy": accuracy_score(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        res.update({f"precision_{avg}": p, f"recall_{avg}": r, f"f1_{avg}": f})
    if y_prob is not None:
        topk = np.argsort(y_prob, axis=1)[:, -k:]
        res[f"top{k}_accuracy"] = float(np.mean([t in row for t, row in zip(y_true, topk)]))
    return res


def print_metrics(res):
    for k, v in res.items():
        print(f"{k:>20}: {v:.4f}")


def per_class_report(y_true, y_pred, class_names=None):
    return classification_report(y_true, y_pred, target_names=class_names, zero_division=0)


def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=True, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(1, keepdims=True).clip(min=1)
    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(max(8, n * 0.18), max(8, n * 0.18)))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")
    ax.set_title("Confusion Matrix" + (" (normalize)" if normalize else ""))
    if class_names is not None and n <= 40:
        ax.set_xticks(range(n)); ax.set_xticklabels(class_names, rotation=90, fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels(class_names, fontsize=7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return cm


def most_confused_pairs(y_true, y_pred, class_names=None, top=15):
    """En sık karıştırılan (gerçek, tahmin) çiftleri."""
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)
    idx = np.dstack(np.unravel_index(np.argsort(cm.ravel())[::-1], cm.shape))[0][:top]
    name = (lambda i: class_names[i]) if class_names else str
    return [(name(i), name(j), int(cm[i, j])) for i, j in idx if cm[i, j] > 0]


def plot_history(history, save_path=None):
    """history: {'train_loss','val_loss','train_acc','val_acc'} listeleri."""
    ep = range(1, len(history["train_loss"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(ep, history["train_loss"], label="train"); a1.plot(ep, history["val_loss"], label="val")
    a1.set_title("Loss"); a1.set_xlabel("Epoch"); a1.legend()
    a2.plot(ep, history["train_acc"], label="train"); a2.plot(ep, history["val_acc"], label="val")
    a2.set_title("Accuracy"); a2.set_xlabel("Epoch"); a2.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def show_misclassified(dataset, y_true, y_pred, y_prob, class_names=None, n=16, save_path=None):
    """En emin olunarak yanlış tahmin edilen n örneği gösterir.
    dataset: değerlendirmede kullanılan (shuffle edilmemiş) dataset."""
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        print("Hatalı tahmin yok."); return []
    wrong = wrong[np.argsort(-y_prob[wrong].max(1))][:n]
    name = (lambda i: class_names[i]) if class_names else str
    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4))
    for ax in np.array(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.array(axes).ravel(), wrong):
        img = dataset[i][0].permute(1, 2, 0).numpy() * IMAGENET_STD + IMAGENET_MEAN
        ax.imshow(img.clip(0, 1))
        ax.set_title(f"G: {name(y_true[i])}\nT: {name(y_pred[i])} ({y_prob[i].max():.2f})",
                     fontsize=8, color="red")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return wrong.tolist()


@torch.no_grad()
def measure_cpu_latency(model, dataset, n=100, warmup=10, num_threads=1):
    """Yarışma koşullarına yakın ölçüm: CPU, batch=1, gerçek test görüntüleri.
    Kişi 1 ile aynı yöntem (100 görüntü, 10 ısınma, 1 thread).
    Döndürür: (ortalama ms, ortanca ms)."""
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    model = model.cpu().eval()
    samples = [dataset[i][0].unsqueeze(0) for i in range(n)]
    for x in samples[:warmup]:
        model(x)
    times = []
    for x in samples:
        t = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t) * 1000)
    torch.set_num_threads(old_threads)
    return float(np.mean(times)), float(np.median(times))


def model_size_mb(path):
    return os.path.getsize(path) / 1024 ** 2


def competition_score(accuracy, ms_per_image, size_mb, t_max=50, s_max=35):
    """Hocanın formülü: 0.60*Acc + 0.25*Hız + 0.15*Boyut (accuracy 0-1 arası verilir)."""
    acc = accuracy * 100
    speed = min(100, t_max / ms_per_image * 100)
    size = min(100, s_max / size_mb * 100)
    return {"accuracy_score": acc, "speed_score": speed, "size_score": size,
            "final_score": 0.60 * acc + 0.25 * speed + 0.15 * size}


def save_metrics_json(path, model_name, accuracy, ms_per_image, size_mb, device, **extra):
    """results/<model>_metrics.json — takımın ortak şeması (generalization_tests.py --task compare)."""
    data = {"model_name": model_name, "accuracy": round(float(accuracy), 4),
            "inference_ms_per_image": round(ms_per_image, 2), "model_size_mb": round(size_mb, 2),
            "device": device}
    data.update({k: round(v, 4) if isinstance(v, float) else v for k, v in extra.items()})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return data


def count_params(model):
    return sum(p.numel() for p in model.parameters())
