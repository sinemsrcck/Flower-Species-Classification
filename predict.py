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


def predict(image_path, model_name=DEFAULT_MODEL, top_k=5):
    return get_predictor(model_name).predict(image_path, top_k=top_k)


def main():
    parser = argparse.ArgumentParser(
        description="Oxford Flowers 102 unified inference"
    )

    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

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
