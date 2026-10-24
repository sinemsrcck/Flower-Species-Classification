"""Shared CLIP loading / inference helpers used by clip.ipynb and generalization_tests.py."""
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_clip(device=None):
    device = device or get_device()
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor, device


def load_cat_to_name(path="data/cat_to_name.json"):
    with open(path) as f:
        return json.load(f)


def build_class_prompts(cat_to_name):
    """Return (class_ids, prompts) sorted by class id, e.g. ["1",...] / ["a photo of a pink primrose, a type of flower", ...]."""
    class_ids = sorted(cat_to_name.keys(), key=int)
    prompts = [f"a photo of a {cat_to_name[c]}, a type of flower" for c in class_ids]
    return class_ids, prompts


@torch.no_grad()
def encode_text_prompts(model, processor, prompts, device):
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    features = model.get_text_features(**inputs)
    return features / features.norm(dim=-1, keepdim=True)


@torch.no_grad()
def encode_images(model, processor, images, device):
    """images: list of PIL.Image (already loaded) -> normalized image feature tensor."""
    inputs = processor(images=images, return_tensors="pt").to(device)
    features = model.get_image_features(**inputs)
    return features / features.norm(dim=-1, keepdim=True)


def model_size_mb(model):
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return param_bytes / (1024 ** 2)


def list_class_images(split_dir):
    """split_dir like data/test -> list of (Path, class_id_str) across all 1..102 subfolders."""
    split_dir = Path(split_dir)
    items = []
    for class_dir in sorted(split_dir.iterdir(), key=lambda p: int(p.name)):
        for img_path in class_dir.glob("*.jpg"):
            items.append((img_path, class_dir.name))
    return items


def open_image(path):
    return Image.open(path).convert("RGB")
