"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: project
from .image_captioner import ImageCaptioner
from .image_captioners import ClipInterrogator, Promptist


class ImageCaptionerManager(Dict):
    """ Represents a ImageCaptionerManager. """
    def __init__(
        self
    ):
        """ Initializes a ImageCaptionerManager. """
        super(ImageCaptionerManager, self).__init__()

        # Image generators
        self["clip_interrogator"]: ClipInterrogator = ClipInterrogator
        self["promptist"]: Promptist = Promptist

    def __call__(
        self,
        captioner_id: str
    ) -> ImageCaptioner:
        """
        Returns the ImageCaptioner of the specified id.

        Parameters
        ----------
            captioner_id: str
                id of the ImageCaptioner to use

        Returns
        ----------
            ImageCaptioner
                desired ImageCaptioner
        """
        if isinstance(self[captioner_id], type):
            self[captioner_id] = self[captioner_id]()

        # Generates image
        return self[captioner_id]
