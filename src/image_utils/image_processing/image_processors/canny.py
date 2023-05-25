"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""


# IMPORT: data processing
import numpy as np
from controlnet_aux import CannyDetector


# IMPORT: project
from src.image_utils.image_processing.image_processor import ImageProcessor


class CannyProcessor(ImageProcessor):
    """ Represents a CannyProcessor. """

    def __init__(
            self
    ):
        """ Initializes a CannyProcessor. """
        # ----- Mother class ----- #
        super(CannyProcessor, self).__init__()

        # ----- Attributes ----- #
        # Processor
        self._processor = CannyDetector()

    def __call__(
        self,
        image: np.ndarray,
        low_threshold: int = 100,
        high_threshold: int = 200,
    ) -> np.ndarray:
        """
        Parameters
        ----------
            image: np.ndarray
                image to process
            low_threshold: int
                canny's low threshold
            high_threshold: int
                canny's high threshold

        Returns
        ----------
            np.ndarray
                Canny mask
        """
        # Processes the image
        return self._processor(image, low_threshold=low_threshold, high_threshold=high_threshold)
