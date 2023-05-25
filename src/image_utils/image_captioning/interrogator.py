"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
from PIL import Image

# IMPORT: data processing
import torch

# IMPORT: deep learning
from clip_interrogator import Config, Interrogator as ClipInterrogator


class Interrogator:
    """
    Represents a Interrogator.

    Attributes
    ----------
        _model: Interrogator
            model needed to generate captions
    """
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(
            self,
            config: Dict[str, Any]
    ):
        """
        Initializes an Interrogator.

        Parameters
        ----------
            config: Dict[str, Any]
                configuration needed to adjust the program behaviour
        """
        # ----- Attributes ----- #
        model_config: Config = Config()

        model_config.blip_offload = config["blip_offload"]
        model_config.chunk_size = config["chunk_size"]
        model_config.flavor_intermediate_count = config["flavor_intermediate_count"]
        model_config.blip_num_beams = config["blip_num_beams"]

        # Model
        self._model: ClipInterrogator = ClipInterrogator(model_config)

    def __call__(
            self,
            image: Image.Image,
            mode: str = None,
            max_flavor: int = 10
    ) -> str:
        """
        Parameters
        ----------
            image: Image.Image
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
        if mode == "best":
            return self._model.interrogate(image, max_flavors=max_flavor)
        elif mode == "fast":
            return self._model.interrogate_fast(image)

        return self._model.interrogate_classic(image)
