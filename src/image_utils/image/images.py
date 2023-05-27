"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: dataset processing
import numpy as np

# IMPORT: project
from .image import Image


class Mask(Image):
    """ Represents a Mask. """
    def __init__(
        self,
        image: Any
    ):
        """
        Initializes a Mask.

        Parameters
        ----------
            image: Any
                image to load
        """
        # ----- Mother class ----- #
        super(Mask, self).__init__(image)

        # ----- Attributes ----- #
        self.image = np.array(self.image)


class ImageToDescribe(Image):
    """ Represents an ImageToDescribe. """
    def __init__(
        self,
        image: Any
    ):
        """
        Initializes an ImageToDescribe.

        Parameters
        ----------
            image: Any
                image to load
        """
        # ----- Mother class ----- #
        super(ImageToDescribe, self).__init__(image)

        # ----- Attributes ----- #
        self.caption: str = ""
        self.improved_caption: str = ""


class ImageToProcess(Image):
    """ Represents an ImageToProcess. """
    def __init__(
        self,
        image: Any
    ):
        """
        Initializes an ImageToProcess.

        Parameters
        ----------
            image: Any
                image to load
        """
        # ----- Mother class ----- #
        super(ImageToProcess, self).__init__(image)
        self.image = np.array(self.image)

        # ----- Attributes ----- #
        self.process_id: str = ""
        self.mask: np.ndarray = np.zeros_like(self.image)

    def reset(self):
        """ Resets the modified image. """
        self.process_id = ""
        self.mask = np.zeros_like(self.image)
