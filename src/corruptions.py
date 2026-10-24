"""Image corruption functions for robustness testing.

Each function takes a PIL.Image and an integer severity in [1, 5] (1 = mild,
5 = severe) and returns a corrupted PIL.Image. Used to simulate the kind of
degraded photos a hidden test set might contain (bad lighting, motion blur,
compression artifacts, partial occlusion, etc.).
"""
import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def gaussian_noise(img: Image.Image, severity: int) -> Image.Image:
    sigma = [5, 10, 20, 35, 55][severity - 1]
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def gaussian_blur(img: Image.Image, severity: int) -> Image.Image:
    radius = [1, 2, 3, 5, 8][severity - 1]
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def brightness(img: Image.Image, severity: int) -> Image.Image:
    factor = [0.8, 0.6, 0.4, 0.25, 0.1][severity - 1]
    return ImageEnhance.Brightness(img).enhance(factor)


def contrast(img: Image.Image, severity: int) -> Image.Image:
    factor = [0.7, 0.5, 0.35, 0.2, 0.1][severity - 1]
    return ImageEnhance.Contrast(img).enhance(factor)


def rotation(img: Image.Image, severity: int) -> Image.Image:
    angle = [5, 10, 20, 35, 50][severity - 1]
    return img.rotate(angle, expand=False, fillcolor=(255, 255, 255))


def jpeg_compression(img: Image.Image, severity: int) -> Image.Image:
    quality = [80, 60, 40, 20, 10][severity - 1]
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def occlusion(img: Image.Image, severity: int) -> Image.Image:
    """Blacks out a random square patch covering an increasing fraction of the image."""
    frac = [0.05, 0.1, 0.2, 0.3, 0.4][severity - 1]
    arr = np.array(img).copy()
    h, w = arr.shape[:2]
    bh, bw = int(h * frac), int(w * frac)
    y = np.random.randint(0, max(h - bh, 1))
    x = np.random.randint(0, max(w - bw, 1))
    arr[y:y + bh, x:x + bw] = 0
    return Image.fromarray(arr)


CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "gaussian_blur": gaussian_blur,
    "brightness": brightness,
    "contrast": contrast,
    "rotation": rotation,
    "jpeg_compression": jpeg_compression,
    "occlusion": occlusion,
}

SEVERITIES = [1, 2, 3, 4, 5]
