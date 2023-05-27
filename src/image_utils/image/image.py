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


class Image:
    """ Represents an Image. """
    def __init__(
        self,
        image: Any
    ):
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
        self.image = PIL.Image.open(image).convert("RGB")
