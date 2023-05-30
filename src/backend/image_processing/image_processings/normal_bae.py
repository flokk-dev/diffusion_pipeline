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
from src.backend.image_processing.image_processing import ImageProcessing


class NormalBae(ImageProcessing):
    """ Represents an NormalBae. """
    control_net_id: str = "lllyasviel/sd-controlnet-normal"

    def __init__(
            self
    ):
        """ Initializes an NormalBae. """
        super(NormalBae, self).__init__()

        # ----- Attributes ----- #
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
        return self._resize(
            self._processor(input_image=image, return_pil=False),
            shape=image.shape
        )
