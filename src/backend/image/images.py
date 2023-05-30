"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: dataset processing
import numpy as np

# IMPORT: project
from .image import Image


class Mask(Image):
    """ Represents a Mask. """
    def __init__(self):
        """ Initializes a Mask. """
        super(Mask, self).__init__()

        # ----- Attributes ----- #
        self.processing: str = ""
        self.weight: float = 1.0

    def reset(self):
        """ Resets the image. """
        super().reset()

        self.processing = ""
        self.weight: float = 1.0


class ImageToDescribe(Image):
    """ Represents an ImageToDescribe. """
    def __init__(self):
        """ Initializes an ImageToDescribe. """
        super(ImageToDescribe, self).__init__()

        # ----- Attributes ----- #
        self.prompt: str = ""

    def reset(self):
        """ Resets the image. """
        super().reset()

        self.prompt = ""


class ImageToProcess(Image):
    """ Represents an ImageToProcess. """
    def __init__(self):
        """ Initializes an ImageToProcess. """
        super(ImageToProcess, self).__init__()

        # ----- Attributes ----- #
        self.processing: str = ""
        self.mask: np.ndarray = np.zeros_like(self.image)

    def load(self, image: Any):
        """
        Loads an Image.

        Parameters
        ----------
            image: Any
                image to load
        """
        super().load(image)

        self.mask = np.zeros_like(self.image)

    def reset(self):
        """ Resets the image. """
        super().reset()

        self.processing = ""
        self.mask = np.zeros_like(self.image)
