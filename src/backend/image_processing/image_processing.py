"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
import numpy as np

# IMPORT: data processing
import cv2
import utils


class ImageProcessing:
    """ Represents an image processing. """

    def __init__(
            self
    ):
        """ Initializes an image processing. """
        # ----- Attributes ----- #
        # Object allowing to process images
        self._processor: Any = None

    def _resize(self, input_image: np.ndarray, output_image: np.ndarray):
        # Retrieves the shape
        h, w = input_image.shape[:2]

        # Resize to the shape of the input image
        return cv2.resize(output_image, (w, h), interpolation=cv2.INTER_AREA)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Runs the image processing into the image.

        Parameters
        ----------
            image: np.ndarray
                image to process

        Returns
        ----------
            np.ndarray
                processed image

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()
