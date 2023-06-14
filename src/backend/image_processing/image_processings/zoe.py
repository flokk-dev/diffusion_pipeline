"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""


# IMPORT: utils
import numpy as np

# IMPORT: deep learning
from controlnet_aux import ZoeDetector

# IMPORT: project
from src.backend.image_processing.image_processing import ImageProcessing


class Zoe(ImageProcessing):
    """ Represents a Zoe processing. """
    control_net_id: str = None

    def __init__(self):
        """ Initializes a Zoe processing. """
        super(Zoe, self).__init__()

        # ----- Attributes ----- #
        # Object allowing to process images
        self._processor: ZoeDetector = ZoeDetector.from_pretrained(
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
        output_image: np.ndarray = self._processor(input_image=image)
        return self._resize(image, output_image)
