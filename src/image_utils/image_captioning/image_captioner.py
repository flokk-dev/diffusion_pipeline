"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: data processing
import torch


class ImageCaptioner:
    """
    Represents an ImageCaptioner.

    Attributes
    ----------
        _model: torch.nn.Module
            model needed to generate the caption of an image.
    """
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self):
        """ Initializes an ImageGenerator. """
        # ----- Attributes ----- #
        # Model
        self._model: torch.nn.Module = None

    def __call__(
        self,
        image: torch.Tensor
    ) -> str:
        """
        Parameters
        ----------
            image: torch.Tensor
                image to generate caption from

        Returns
        ----------
            str
                caption that describes the image

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()
