"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""


# IMPORT: data processing
import numpy as np

# IMPORT: deep learning
from controlnet_aux import OpenposeDetector

# IMPORT: project
from src.image_utils.image_processing.image_processor import ImageProcessor


class OpenPoseProcessor(ImageProcessor):
    """ Represents a OpenPoseProcessor. """
    control_net_id: str = "lllyasviel/sd-controlnet-openpose"

    def __init__(
            self
    ):
        """ Initializes a OpenPoseProcessor. """
        super(OpenPoseProcessor, self).__init__()

        # ----- Attributes ----- #
        self._processor = OpenposeDetector.from_pretrained(
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
                Pose mask
        """
        # Processes the image
        return self._resize(
            self._processor(input_image=image, return_pil=False),
            shape=image.shape
        )
