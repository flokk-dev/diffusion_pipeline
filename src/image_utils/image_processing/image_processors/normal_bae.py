"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""


# IMPORT: data processing
import numpy as np

# IMPORT: deep learning
from controlnet_aux import NormalBaeDetector

# IMPORT: project
from src.image_utils.image_processing.image_processor import ImageProcessor


class NormalBaeProcessor(ImageProcessor):
    """ Represents an NormalBaeProcessor. """

    def __init__(
            self
    ):
        """ Initializes an NormalBaeProcessor. """
        # ----- Mother class ----- #
        super(NormalBaeProcessor, self).__init__()

        # ----- Attributes ----- #
        # Processor
        self._processor = NormalBaeDetector.from_pretrained(
            pretrained_model_or_path="lllyasviel/Annotators"
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
                NormalBAE mask
        """
        # Processes the image
        return self._processor(input_image=image, return_pil=False)
