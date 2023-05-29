"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""


# IMPORT: data processing
import numpy as np

# IMPORT: deep learning
from controlnet_aux import MLSDdetector

# IMPORT: project
from src.image_utils.image_processing.image_processor import ImageProcessor


class MLSDProcessor(ImageProcessor):
    """ Represents an MLSDProcessor. """
    control_net_id: str = "lllyasviel/sd-controlnet-mlsd"

    def __init__(
            self
    ):
        """ Initializes an MLSDProcessor. """
        super(MLSDProcessor, self).__init__()

        # ----- Attributes ----- #
        self._processor = MLSDdetector.from_pretrained(
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
                MLSD mask
        """
        # Processes the image
        return self._resize(
            self._processor(input_image=image, return_pil=False),
            shape=image.shape
        )
