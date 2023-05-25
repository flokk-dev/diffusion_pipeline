"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: project
from .image_captioning import ImageCaptionerManager
from .image_processing import ImageProcessorManager
from .image_generation import ImageGeneratorManager


class Backend:
    """
    Represents a Backend.

    Attributes
    ----------
        image_captioning_manager: ImageCaptionerManager
            manager of the different image captioning method
        image_processing_manager: ImageProcessorManager
            manager of the different image processing method
        image_generation_manager: ImageGeneratorManager
            manager of the different image generation method
    """
    def __init__(
        self
    ):
        """ Initializes a Backend. """
        # ----- Attributes ----- #
        # Image captioning
        self.image_captioning_manager = ImageCaptionerManager()

        # Image processing
        self.image_processing_manager = ImageProcessorManager()

        # Image generation
        self.image_generation_manager = ImageGeneratorManager()
