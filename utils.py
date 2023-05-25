"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from pynvml import *

import datetime

# IMPORT: dataset processing
import PIL
import numpy as np


# ---------- INFO ---------- #

def get_datetime():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)

    return f"{(info.used / 1024**3):.3f}Go"


# ---------- OBJECTS ---------- #

class Image:
    """ Represents an Image. """
    def __init__(
        self,
        image_id: int,
        image_path: str
    ):
        """
        Initializes an Image.

        Parameters
        ----------
            image_id: int
                id of the image
            image_path: str
                path of the image
        """
        # ----- Attributes ----- #
        self.id: int = image_id
        self.path: str = image_path

        # Image
        self.image: np.ndarray = np.array(PIL.Image.open(image_path).convert("RGB"))


class ImageToProcess(Image):
    """ Represents an ImageToProcess. """
    def __init__(
        self,
        image_id: int,
        image_path: str
    ):
        """
        Initializes an ImageToProcess.

        Parameters
        ----------
            image_id: int
                id of the image
            image_path: str
                path of the image
        """
        # ----- Mother class ----- #
        super(ImageToProcess, self).__init__(image_id, image_path)

        # ----- Attributes ----- #
        self.process_id: str = ""
        self.modified_image: np.ndarray = np.zeros_like(self.image)

    def reset(self):
        """ Resets the modified image. """
        self.process_id = ""
        self.modified_image = np.zeros_like(self.image)
