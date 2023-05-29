"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""


# IMPORT: data processing
import numpy as np

# IMPORT: deep learning
from controlnet_aux import ZoeDetector

# IMPORT: project
from src.image_utils.image_processing.image_processor import ImageProcessor


class ZoeProcessor(ImageProcessor):
    """ Represents an ZoeProcessor. """
    control_net_id: str = None

    def __init__(
            self
    ):
        """ Initializes an ZoeProcessor. """
        super(ZoeProcessor, self).__init__()

        # ----- Attributes ----- #
        # Processor
        self._processor = ZoeDetector.from_pretrained(
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
                PidiNet mask
        """
        # Processes the image
        return self._resize(
            self._processor(input_image=image, return_pil=False),
            shape=image.shape
        )
