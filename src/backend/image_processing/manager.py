"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
import numpy as np

# IMPORT: project
from .image_processing import ImageProcessing
from .image_processings import *


class ImageProcessingManager(Dict):
    """ Represents the object allowing to manage multiple image processing. """
    def __init__(self):
        """ Initializes the object allowing to manage multiple image processing. """
        super(ImageProcessingManager, self).__init__()

        # Adds all the processing to the manager
        self["canny"]: Canny = Canny
        self["hed"]: Hed = Hed
        self["lineart"]: Lineart = Lineart
        self["lineart anime"]: LineartAnime = LineartAnime
        self["mediapipe face"]: MediapipeFace = MediapipeFace
        self["midas"]: Midas = Midas
        self["mlsd"]: MLSD = MLSD
        self["normal"]: NormalBae = NormalBae
        self["pidi"]: PidiNet = PidiNet
        self["openpose"]: OpenPose = OpenPose
        self["sam"]: Sam = Sam
        self["shuffle"]: ContentShuffle = ContentShuffle
        self["zoe"]: Zoe = Zoe

    def __call__(self, processing_id: str, image: np.ndarray) -> np.ndarray:
        """
        Runs the processing linked to the id into the image.

        Parameters
        ----------
            processing_id: str
                id of an image processing
            image: np.ndarray
                image to process

        Returns
        ----------
            np.ndarray
                processed image
        """
        # Runs the processing into the image
        return self[processing_id]()(image=image)
