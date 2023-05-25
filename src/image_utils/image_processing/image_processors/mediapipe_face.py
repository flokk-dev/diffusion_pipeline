"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""


# IMPORT: data processing
import numpy as np

# IMPORT: deep learning
from controlnet_aux import MediapipeFaceDetector

# IMPORT: project
from src.image_utils.image_processing.image_processor import ImageProcessor


class MediapipeFaceProcessor(ImageProcessor):
    """ Represents an MediapipeFaceProcessor. """

    def __init__(
            self
    ):
        """ Initializes an MediapipeFaceProcessor. """
        # ----- Mother class ----- #
        super(MediapipeFaceProcessor, self).__init__()

        # ----- Attributes ----- #
        # Processor
        self._processor = MediapipeFaceDetector()

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
        return self._processor(image=image, return_pil=False)