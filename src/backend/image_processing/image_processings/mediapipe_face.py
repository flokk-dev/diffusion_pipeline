"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""


# IMPORT: utils
import numpy as np

# IMPORT: deep learning
from controlnet_aux import MediapipeFaceDetector

# IMPORT: project
from src.backend.image_processing.image_processing import ImageProcessing


class MediapipeFace(ImageProcessing):
    """ Represents a MediapipeFace processing. """

    def __init__(
            self
    ):
        """ Initializes a MediapipeFace processing. """
        super(MediapipeFace, self).__init__()

        # ----- Attributes ----- #
        # Object allowing to process images
        self._processor: MediapipeFaceDetector = MediapipeFaceDetector()

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
        return self._processor(image=image, return_pil=False)
