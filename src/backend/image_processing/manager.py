"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: project
from .image_processing import ImageProcessing
from .image_processings import *


class ImageProcessingManager(Dict):
    """ Represents a ImageProcessingManager. """
    def __init__(self):
        """ Initializes a Backend. """
        super(ImageProcessingManager, self).__init__()

        # Image processor
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

    def __call__(
        self,
        process_id: str
    ) -> ImageProcessing:
        """
        Returns the ImageProcessing of the specified id.

        Parameters
        ----------
            process_id: str
                id of the ImageProcessing to use

        Returns
        ----------
            ImageProcessing
                desired ImageProcessing
        """
        if isinstance(self[process_id], type):
            self[process_id] = self[process_id]()

        # Processes image
        return self[process_id]
