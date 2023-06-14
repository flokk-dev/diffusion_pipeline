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
        self["Canny"]: Canny = Canny
        self["Depth"]: Midas = Midas
        self["Hed"]: Hed = Hed
        self["Lineart"]: Lineart = Lineart
        self["Lineart anime"]: LineartAnime = LineartAnime
        # self["mediapipe face"]: MediapipeFace = MediapipeFace
        self["MLSD"]: MLSD = MLSD
        self["Normal BAE"]: NormalBae = NormalBae
        # self["pidi"]: PidiNet = PidiNet
        self["OpenPose"]: OpenPose = OpenPose
        self["Seg"]: Sam = Sam
        self["Shuffle"]: ContentShuffle = ContentShuffle
        # self["zoe"]: Zoe = Zoe

    def __call__(self, image: np.ndarray, processing_id: str) -> np.ndarray:
        """
        Runs the processing linked to the id into the image.

        Parameters
        ----------
            image: np.ndarray
                image to process
            processing_id: str
                id of an image processing

        Returns
        ----------
            np.ndarray
                processed image
        """
        if isinstance(self[processing_id], type):
            self[processing_id] = self[processing_id]()

        # Runs the processing into the image
        return self[processing_id](image=image)
