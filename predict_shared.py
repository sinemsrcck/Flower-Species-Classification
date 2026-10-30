import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from shared_model_loader import SharedModel

PROJECT_DIR = Path(__file__).resolve().parent
CAT_TO_NAME_PATH = PROJECT_DIR / "cat_to_name.json"


def load_cat_to_name():
    with open(CAT_TO_NAME_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


class SharedPredictor:
    def __init__(self, model_name):
        self.model = SharedModel(model_name, device="cpu")
        self.transform = get_transform()
        # model.class_to_idx: {"1": 0, "2": 1, ...} -> idx_to_class_id: {0: "1", 1: "2", ...}
        self.idx_to_class_id = {
            index: class_id
            for class_id, index in self.model.class_to_idx.items()
        }
        self.cat_to_name = load_cat_to_name()

    def predict(self, image_path, top_k=5):
        image_path = Path(image_path)

        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        if not 1 <= top_k <= self.model.num_classes:
            raise ValueError("Invalid top_k value.")

        with Image.open(image_path) as image:
            tensor = self.transform(
                image.convert("RGB")
            ).unsqueeze(0)

        logits = self.model.predict_logits(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

        scores, indices = torch.topk(
            probabilities,
            k=top_k
        )

        predictions = []

        for score, index in zip(scores, indices):
            class_index = int(index.item())
            class_id = self.idx_to_class_id[class_index]

            predictions.append({
                "class_name": self.cat_to_name.get(class_id, f"unknown (class {class_id})"),
                "class_id": class_id,
                "class_index": class_index,
                "confidence": float(score.item())
            })

        return {
            "model": self.model.name,
            "image": str(image_path),
            "predictions": predictions
        }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    predictor = SharedPredictor(args.model)

    result = predictor.predict(
        image_path=args.image,
        top_k=args.top_k
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)

    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
