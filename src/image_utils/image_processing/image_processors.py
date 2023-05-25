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
from controlnet_aux import OpenposeDetector

# IMPORT: project
from .image_processor import ImageProcessor


class Canny(ImageProcessor):
    """ Represents a Canny. """

    def __init__(
            self
    ):
        """ Initializes a Canny. """
        # ----- Mother Class ----- #
        super(Canny, self).__init__()

    def __call__(
        self,
        image: np.ndarray,
        low_threshold: int = 100,
        high_threshold: int = 200,
    ) -> np.ndarray:
        """
        Parameters
        ----------
            image: np.ndarray
                image to pre-process
            low_threshold: int
                canny's low threshold
            high_threshold: int
                canny's high threshold

        Returns
        ----------
            np.ndarray
                canny mask
        """
        # Applies canny filter
        mask: np.ndarray = cv2.Canny(image, low_threshold, high_threshold)

        # Adjusts the shape
        mask: np.ndarray = mask[:, :, None]
        return np.concatenate([mask, mask, mask], axis=2)


class Pose(ImageProcessor):
    """ Represents a Pose. """

    def __init__(
            self
    ):
        """ Initializes a Pose. """
        # ----- Mother Class ----- #
        super(Pose, self).__init__()

        # ----- Attributes ----- #
        # Model
        self._model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

    def __call__(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Parameters
        ----------
            image: np.ndarray
                image to pre-process

        Returns
        ----------
            np.ndarray
                pose mask
        """
        # Retrieves width and height from the original image
        w, h = image.shape[1], image.shape[0]

        # Detects the pose
        mask = np.array(self._model(image))
        return cv2.resize(mask, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
