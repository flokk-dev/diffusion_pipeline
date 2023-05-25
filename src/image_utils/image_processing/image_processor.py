"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: data processing
import numpy as np


class ImageProcessor:
    """ Represents a ImageProcessor. """

    def __init__(
            self
    ):
        """ Initializes a ImageProcessor. """
        # ----- Attributes ----- #
        self._width: int = 1920
        self._height: int = 1080

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
                output mask

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()
