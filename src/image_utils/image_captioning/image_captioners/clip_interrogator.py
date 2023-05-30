"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import PIL

# IMPORT: deep learning
from clip_interrogator import Config, Interrogator

# IMPORT: project
from src.backend.image_captioning.image_captioner import ImageCaptioner


class ClipInterrogator(ImageCaptioner):
    """
    Represents a ClipInterrogator.

    Attributes
    ----------
        _model: Interrogator
            model needed to generate captions
    """
    def __init__(self):
        """ Initializes a ClipInterrogator. """
        super(ClipInterrogator, self).__init__()

        # ----- Attributes ----- #
        model_config: Config = Config()

        model_config.blip_offload = True
        model_config.chunk_size = 2048
        model_config.flavor_intermediate_count = 512
        model_config.blip_num_beams = 64

        # Model
        self._model: Interrogator = Interrogator(model_config)
        self._model.device = "cpu"

    def __call__(
        self,
        image: PIL.Image,
        mode: str = None,
        max_flavor: int = 10
    ) -> str:
        """
        Parameters
        ----------
            image: PIL.Image
                image to generate caption from
            mode: str
                mode of the captioning
            max_flavor: int
                ...

        Returns
        ----------
            str
                caption that describes the image
        """
        image = PIL.Image.fromarray(image)

        self._model.device = "cuda"
        if mode == "best":
            image = self._model.interrogate(image, max_flavors=max_flavor)
        elif mode == "fast":
            image = self._model.interrogate_fast(image)
        else:
            image = self._model.interrogate_classic(image)

        self._model.device = "cpu"
        return image
