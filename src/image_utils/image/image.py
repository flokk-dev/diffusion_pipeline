"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: dataset loading
import PIL

# IMPORT: dataset processing
import numpy as np


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
        self.image = PIL.Image.open(image_path).convert("RGB")
