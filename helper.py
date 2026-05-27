from PIL import Image
import cv2
import numpy as np


def color_to_gray(img: Image.Image) -> np.ndarray:
    """Convert a PIL image to an 8-bit grayscale NumPy array."""
    return np.array(img.convert("L"))


def resize_gray(img: np.ndarray, max_dim: int = 320) -> np.ndarray:
    """Down-scale if either dimension exceeds max_dim (keeps aspect ratio)."""
    h, w = img.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img