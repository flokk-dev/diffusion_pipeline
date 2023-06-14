"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: data processing
import cv2
import numpy as np
import torch.cuda

# IMPORT: deep learning
from controlnet_aux import MidasDetector

# IMPORT: project
from src.backend.image_processing.image_processing import ImageProcessing


class Midas(ImageProcessing):
    """ Represents a Midas processing. """

    def __init__(
            self
    ):
        """ Initializes a Midas processing. """
        super(Midas, self).__init__()

        # ----- Attributes ----- #
        # Object allowing to process images
        self._processor: MidasDetector = MidasDetector.from_pretrained(
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
        # Resizes the shape in order to make the Midas processing work
        output_image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LANCZOS4)

        # Runs the processing into the image
        output_image = self._processor(input_image=output_image)

        # Resizes the output image to its original shape
        output_image: np.ndarray = np.stack((output_image, output_image, output_image), axis=2)
        return self._resize(image, output_image)
