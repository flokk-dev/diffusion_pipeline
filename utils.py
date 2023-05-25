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

# IMPORT: dataset processing
import cv2
import numpy as np


# ---------- INFO ---------- #

def get_datetime():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)

    return f"{(info.used / 1024 ** 3):.3f}Go"


# ---------- DATA PROCESSING ---------- #
def resize_to_shape(image: np.ndarray, shape: Tuple[int]) -> np.ndarray:
    return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_LANCZOS4)


def resize_image(image: np.ndarray, resolution: int):
    H, W, C = image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64

    return cv2.resize(
        image,
        (W, H),
        interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA
    )
