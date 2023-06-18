"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
from pynvml import *

import datetime
import re

# IMPORT: dataset processing
import cv2
import numpy as np

from PIL import Image
import torch


# ---------- INFO ---------- #

def get_datetime():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)

    return f"{(info.used / 1024 ** 3):.3f}Go"


# ---------- ... ---------- #
def get_key_words(text: str):
    matches = re.findall(r"'(.*?)'", text)
    if not matches:
        return None

    words = [match.split() for match in matches]
    return words[0][0]

# ---------- DATA PROCESSING ---------- #

def resize_image(image: np.ndarray, resolution: int):
    h, w = image.shape[:2]
    h = float(h)
    w = float(w)
    k = float(resolution) / min(h, w)
    h *= k
    w *= k
    h = int(np.round(h / 8.0)) * 8
    w = int(np.round(w / 8.0)) * 8

    return cv2.resize(
        image,
        (w, h),
        interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA
    )


def tensor_to_image(tensor: torch.Tensor):
    images = list()
    for image in tensor:
        # From [-1, 1] to [0, 1]
        image = (image / 2 + 0.5).clamp(0, 1).unsqueeze(0)

        # To numpy ndarray
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]

        # To PIL Image
        image = Image.fromarray((image * 255).round().astype("uint8")).convert('RGB')
        images.append(image)

    return images
