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
    def __init__(self):
        """ Initializes an Image. """
        # ----- Attributes ----- #
        self.id: int = None
        self.name: str = "none.jpg"

        # Image
        self.image = np.zeros((480, 640, 3))

    def load(self, image: Any):
        """
        Loads an Image.

        Parameters
        ----------
            image: Any
                image to load
        """
        self.id: int = image.id
        self.name: str = image.name

        # Converts the image into an array
        self.image = np.array(PIL.Image.open(image).convert("RGB"))

    def reset(self):
        """ Resets the image. """
        self.id: int = None
        self.name: str = "none.jpg"

        # Image
        self.image = np.zeros((480, 640, 3))


class Images(list):
    """ Represents an Images. """
    def __init__(
        self,
        image_type: type,
        n_images: int = 3
    ):
        """
        Initializes an Images.

        Parameters
        ----------
            image_type: type
                type of the images
            n_images: str
                number of images to store
        """
        super(Images, self).__init__()

        # Creates a list of images
        for _ in range(n_images):
            # Appends an image of the desired type
            self.append(image_type())

    def __len__(self):
        length = 0

        # For each image in the list
        for image in self:
            # If the id of the image is not None
            if image.id is not None:
                # Increments by one the length of the list
                length += 1

        return length
