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

    def __init__(
            self
    ):
        """ Initializes a OpenPoseProcessor. """
        # ----- Mother class ----- #
        super(OpenPoseProcessor, self).__init__()

        # ----- Attributes ----- #
        # Processor
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
        return self._processor(input_image=image, hand_and_face=True, return_pil=False)