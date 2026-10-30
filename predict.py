import argparse
import json
from pathlib import Path

from predict_shared import SharedPredictor


DEFAULT_MODEL = "efficientnet_b0_onnx"

# Model her çağrıda yeniden yüklenmesin diye önbellekte tutulur
# (döngü içinde predict() çağrıldığında süreye model yükleme eklenmez).
_PREDICTORS = {}


def get_predictor(model_name=DEFAULT_MODEL):
    if model_name not in _PREDICTORS:
        _PREDICTORS[model_name] = SharedPredictor(model_name)
    return _PREDICTORS[model_name]


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def predict(image_path, model_name=DEFAULT_MODEL, top_k=5):
    return get_predictor(model_name).predict(image_path, top_k=top_k)


def predict_dir(image_dir, model_name=DEFAULT_MODEL, top_k=5):
    """Klasördeki (alt klasörler dahil) tüm görüntüleri tahmin eder.
    Görüntüler sınıf numarası adlı alt klasörlerdeyse (ör. test/47/x.jpg)
    top-1 accuracy de hesaplanır."""
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise NotADirectoryError(image_dir)

    paths = sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No images found in: {image_dir}")

    predictor = get_predictor(model_name)
    results, correct, labeled = [], 0, 0
    for path in paths:
        result = predictor.predict(path, top_k=top_k)
        true_id = path.parent.name
        if true_id in predictor.model.class_to_idx:
            labeled += 1
            correct += result["predictions"][0]["class_id"] == true_id
        results.append(result)

    summary = {"model": model_name, "num_images": len(paths), "results": results}
    if labeled:
        summary["accuracy"] = correct / labeled
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Oxford Flowers 102 unified inference"
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Tek görüntü yolu")
    source.add_argument("--image-dir", help="Görüntü klasörü (alt klasörler dahil)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    if args.image_dir:
        result = predict_dir(args.image_dir, model_name=args.model, top_k=args.top_k)
    else:
        result = predict(
            image_path=args.image,
            model_name=args.model,
            top_k=args.top_k
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )
        output_path.write_text(
            json.dumps(result, indent=4),
            encoding="utf-8"
        )

    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
