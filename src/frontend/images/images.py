"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT utils
from typing import *

# IMPORT: dataset loading
import PIL

# IMPORT: data processing
import numpy as np


class Image:
    """ Represents an Image. """
    def __init__(self, image: Any):
        """
        Initializes an Image.

        Parameters
        ----------
            image: Any
                image to load
        """
        # ----- Attributes ----- #
        self.id: int = image.id
        self.name: str = image.name

        # Image
        self.image = np.array(PIL.Image.open(image).convert("RGB"))


class Mask(Image):
    """ Represents a Mask. """
    def __init__(self, image: Any):
        """
        Initializes a Mask.

        Parameters
        ----------
            image: Any
                image to load
        """
        super(Mask, self).__init__(image=image)

        # ----- Attributes ----- #
        self.processing: str = ""
        self.weight: float = 1.0


class ImageToDescribe(Image):
    """ Represents an ImageToDescribe. """
    def __init__(self, image: Any):
        """
        Initializes an ImageToDescribe.

        Parameters
        ----------
            image: Any
                image to load
        """
        super(ImageToDescribe, self).__init__(image=image)

        # ----- Attributes ----- #
        self.prompt: str = ""


class ImageToProcess(Image):
    """ Represents an ImageToProcess. """
    def __init__(self, image: Any):
        """
        Initializes an ImageToProcess.

        Parameters
        ----------
            image: Any
                image to load
        """
        super(ImageToProcess, self).__init__(image=image)

        # ----- Attributes ----- #
        self.processing: str = ""
        self.mask: np.ndarray = np.zeros_like(self.image)
