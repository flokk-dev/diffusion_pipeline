"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from PIL import Image

# IMPORT: deep learning
from clip_interrogator import Config, Interrogator


class ClipInterrogator:
    """
    Represents an object allowing to transform images into prompts.
    Attributes
    ----------
        _model: Interrogator
            model allowing to transform images into prompts
    """
    def __init__(self):
        """ Initializes an object allowing to transform images into prompts. """
        # ----- Attributes ----- #
        # ClipInterrogator configuration
        model_config: Config = Config()

        model_config.blip_offload = True
        model_config.chunk_size = 2048
        model_config.flavor_intermediate_count = 512
        model_config.blip_num_beams = 64

        # Model allowing to transform images into prompts
        self._model: Interrogator = Interrogator(model_config)

    def __call__(self, image: Image.Image, mode: str = None, max_flavor: int = 10) -> str:
        """
        Parameters
        ----------
            image: PIL.Image
                image to transform into a prompt
            mode: str
                interrogation mode
            max_flavor: int
                ...

        Returns
        ----------
            str
                prompt version of the image
        """
        # If the image is not a PIL image
        image = Image.fromarray(image)

        # Transforms the image into a prompt depending on the interrogation mode
        if mode == "best":
            return self._model.interrogate(image, max_flavors=max_flavor)
        elif mode == "fast":
            return self._model.interrogate_fast(image)
        else:
            return self._model.interrogate_classic(image)
