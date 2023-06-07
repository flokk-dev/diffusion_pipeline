"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""


# IMPORT: utils
import numpy as np

# IMPORT: deep learning
from controlnet_aux import HEDdetector

# IMPORT: project
from src.backend.image_processing.image_processing import ImageProcessing


class Hed(ImageProcessing):
    """ Represents an HED processing. """

    def __init__(
            self
    ):
        """ Initializes an HED processing. """
        super(Hed, self).__init__()

        # ----- Attributes ----- #
        # Object allowing to process images
        self._processor: HEDdetector = HEDdetector.from_pretrained(
            pretrained_model_or_path="lllyasviel/Annotators"
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Runs the processing into the image.

        Parameters
        ----------
            image: np.ndarray
                image to process

        Returns
        ----------
            np.ndarray
                processed image
        """
        # Runs the processing into the image
        return self._processor(input_image=image, return_pil=False)