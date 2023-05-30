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
from src.backend.image_processing.image_processor import ImageProcessor


class SamProcessor(ImageProcessor):
    """ Represents an SamProcessor. """
    control_net_id: str = "lllyasviel/sd-controlnet-seg"

    def __init__(
            self
    ):
        """ Initializes an SamProcessor. """
        super(SamProcessor, self).__init__()

        # ----- Attributes ----- #
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
        return self._resize(
            np.array(self._processor(image=image)),
            shape=image.shape
        )
