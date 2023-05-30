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
from src.backend.image_processing.image_processing import ImageProcessing


class ContentShuffle(ImageProcessing):
    """ Represents a ContentShuffle. """
    control_net_id: str = None

    def __init__(
            self
    ):
        """ Initializes a ContentShuffle. """
        super(ContentShuffle, self).__init__()

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
        return self._resize(
            self._processor(input_image=image, return_pil=False),
            shape=image.shape
        )
