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
from src.backend.image_processing.image_processing import ImageProcessing


class Sam(ImageProcessing):
    """ Represents a Sam processing. """

    def __init__(self):
        """ Initializes  a Sam processing. """
        super(Sam, self).__init__()

        # ----- Attributes ----- #
        # Object allowing to process images
        self._processor: SamDetector = SamDetector.from_pretrained(
            pretrained_model_or_path="segments-arnaud/sam_vit_h"
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
        output_image: np.ndarray = np.array(self._processor(image=image))
        return self._resize(image, output_image)
