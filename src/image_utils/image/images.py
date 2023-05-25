"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: dataset processing
import numpy as np

# IMPORT: project
from .image import Image


class Mask(Image):
    """ Represents a Mask. """
    def __init__(
        self,
        image_id: int,
        image_path: str
    ):
        """
        Initializes a Mask.

        Parameters
        ----------
            image_id: int
                id of the image
            image_path: str
                path of the image
        """
        # ----- Mother class ----- #
        super(Mask, self).__init__(image_id, image_path)

        # ----- Attributes ----- #
        self.image = np.array(self.image)


class ImageToDescribe(Image):
    """ Represents an ImageToDescribe. """
    def __init__(
        self,
        image_id: int,
        image_path: str
    ):
        """
        Initializes an ImageToDescribe.

        Parameters
        ----------
            image_id: int
                id of the image
            image_path: str
                path of the image
        """
        # ----- Mother class ----- #
        super(ImageToDescribe, self).__init__(image_id, image_path)

        # ----- Attributes ----- #
        self.caption: str = ""


class ImageToProcess(Image):
    """ Represents an ImageToProcess. """
    def __init__(
        self,
        image_id: int,
        image_path: str
    ):
        """
        Initializes an ImageToProcess.

        Parameters
        ----------
            image_id: int
                id of the image
            image_path: str
                path of the image
        """
        # ----- Mother class ----- #
        super(ImageToProcess, self).__init__(image_id, image_path)

        # ----- Attributes ----- #
        self.process_id: str = ""
        self.modified_image: np.ndarray = np.zeros_like(self.image)

    def reset(self):
        """ Resets the modified image. """
        self.process_id = ""
        self.modified_image = np.zeros_like(self.image)
