"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""


# IMPORT: data processing
import numpy as np

# IMPORT: deep learning
from controlnet_aux import ContentShuffleDetector

# IMPORT: project
from src.image_utils.image_processing.image_processor import ImageProcessor


class ContentShuffleProcessor(ImageProcessor):
    """ Represents a ContentShuffleProcessor. """

    def __init__(
            self
    ):
        """ Initializes a ContentShuffleProcessor. """
        # ----- Mother class ----- #
        super(ContentShuffleProcessor, self).__init__()

        # ----- Attributes ----- #
        # Processor
        self._processor = ContentShuffleDetector()

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
                ContentShuffle mask
        """
        # Processes the image
        return self._processor(input_image=image, return_pil=False)
