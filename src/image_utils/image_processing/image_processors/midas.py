"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: data processing
import cv2
import numpy as np

# IMPORT: deep learning
from controlnet_aux import MidasDetector

# IMPORT: project
import utils

from src.image_utils.image_processing.image_processor import ImageProcessor


class MidasProcessor(ImageProcessor):
    """ Represents an MidasProcessor. """
    control_net_id: str = "lllyasviel/sd-controlnet-depth"

    def __init__(
            self
    ):
        """ Initializes an MidasProcessor. """
        super(MidasProcessor, self).__init__()

        # ----- Attributes ----- #
        self._processor = MidasDetector.from_pretrained(
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
                Midas mask
        """
        # Modify shape to work with midas
        image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LANCZOS4)

        # Processes the image
        depth_map = self._processor(input_image=image)

        # Resizes the depth_map
        depth_map = np.stack((depth_map, depth_map, depth_map), axis=2)
        return self._resize(image=depth_map, shape=image.shape)
