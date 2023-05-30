"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: data processing
import cv2
import numpy as np


class ImageProcessing:
    """ Represents a ImageProcessing. """
    control_net_id: str = None

    def __init__(
            self
    ):
        """ Initializes a ImageProcessing. """
        # ----- Attributes ----- #
        # Object needed to process images
        self._processor: Any = None

    @staticmethod
    def _resize(
        image: np.ndarray,
        shape: Tuple[int]
    ) -> np.ndarray:
        """
        Resizes an image to a given shape.

        Parameters
        ----------
            image: np.ndarray
                image to resize
            shape: Tuple[int]
                desired shape

        Returns
        ----------
            np.ndarray
                resized image
        """
        return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_LANCZOS4)

    def __call__(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Parameters
        ----------
            image: np.ndarray
                image to process

        Returns
        ----------
            np.ndarray
                output mask

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()
