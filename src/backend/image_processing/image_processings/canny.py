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
from controlnet_aux import CannyDetector


# IMPORT: project
from src.backend.image_processing.image_processing import ImageProcessing


class Canny(ImageProcessing):
    """ Represents a Canny processing. """

    def __init__(self):
        """ Initializes a Canny processing. """
        super(Canny, self).__init__()

        # ----- Attributes ----- #
        # Object allowing to process images
        self._processor: CannyDetector = CannyDetector()

    def __call__(self, image: np.ndarray, thresholds: Tuple[int] = (100, 200)) -> np.ndarray:
        """
        Runs the processing into the image.

        Parameters
        ----------
            image: np.ndarray
                image to process
            thresholds: Tuple[int]
                low and high canny threshold

        Returns
        ----------
            np.ndarray
                processed image
        """
        # Runs the processing into the image
        output_image: np.ndarray = self._processor(
            img=image,
            low_threshold=thresholds[0], high_threshold=thresholds[1]
        )

        # Resizes the output image to its original shape
        return self._resize(image=output_image, shape=image.shape)
