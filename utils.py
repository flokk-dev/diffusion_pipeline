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

def resize_image(image: np.ndarray, resolution: int):
    h, w = image.shape[:2]
    h = float(h)
    w = float(w)
    k = float(resolution) / min(h, w)
    h *= k
    w *= k
    h = int(np.round(h / 8.0)) * 8
    w = int(np.round(w / 8.0)) * 8
    print(h, w)

    return cv2.resize(
        image,
        (w, h),
        interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA
    )
