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


class ImageProcessing:
    """ Represents an image processing. """

    def __init__(
            self
    ):
        """ Initializes an image processing. """
        # ----- Attributes ----- #
        # Object allowing to process images
        self._processor: Any = None

    @staticmethod
    def _resize(image: np.ndarray, shape: Tuple[int]) -> np.ndarray:
        """
        Resizes an image.

        Parameters
        ----------
            image: np.ndarray
                image to resize
            shape: Tuple[int]
                output shape

        Returns
        ----------
            np.ndarray
                resized image
        """
        return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_LANCZOS4)

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
