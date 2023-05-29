"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: project
from .image_processor import ImageProcessor
from .image_processors import *


class ImageProcessorManager(Dict):
    """ Represents a ImageProcessorManager. """
    def __init__(self):
        """ Initializes a Backend. """
        super(ImageProcessorManager, self).__init__()

        # Image processors
        self["canny"]: CannyProcessor = CannyProcessor
        self["hed"]: HedProcessor = HedProcessor
        self["lineart"]: LineartProcessor = LineartProcessor
        self["lineart anime"]: LineartAnimeProcessor = LineartAnimeProcessor
        self["mediapipe face"]: MediapipeFaceProcessor = MediapipeFaceProcessor
        self["midas"]: MidasProcessor = MidasProcessor
        self["mlsd"]: MLSDProcessor = MLSDProcessor
        self["normal"]: NormalBaeProcessor = NormalBaeProcessor
        self["pidi"]: PidiNetProcessor = PidiNetProcessor
        self["openpose"]: OpenPoseProcessor = OpenPoseProcessor
        self["sam"]: SamProcessor = SamProcessor
        self["shuffle"]: ContentShuffleProcessor = ContentShuffleProcessor
        self["zoe"]: ZoeProcessor = ZoeProcessor

    def __call__(
        self,
        process_id: str
    ) -> ImageProcessor:
        """
        Returns the ImageProcessor of the specified id.

        Parameters
        ----------
            process_id: str
                id of the ImageProcessor to use

        Returns
        ----------
            ImageProcessor
                desired ImageProcessor
        """
        if isinstance(self[process_id], type):
            self[process_id] = self[process_id]()

        # Processes image
        return self[process_id]
