"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""


# IMPORT: data processing
import numpy as np

# IMPORT: deep learning
from controlnet_aux import SamDetector

# IMPORT: project
from src.image_utils.image_processing.image_processor import ImageProcessor


class SamProcessor(ImageProcessor):
    """ Represents an SamProcessor. """

    def __init__(
            self
    ):
        """ Initializes an SamProcessor. """
        # ----- Mother class ----- #
        super(SamProcessor, self).__init__()

        # ----- Attributes ----- #
        # Processor
        self._processor = SamDetector.from_pretrained(
            pretrained_model_or_path="segments-arnaud/sam_vit_h"
        )

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
                PidiNet mask
        """
        # Processes the image
        return np.array(
            self._processor(image=image)
        )
