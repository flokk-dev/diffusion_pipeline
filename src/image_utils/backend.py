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
import numpy as np
import torchvision

# IMPORT: deep learning

# IMPORT: project
from src.image_utils import image_processing
from src.image_utils.image_diffusion import \
    DiffusionPipeline, StableDiffusionPipeline, ControlDiffusionPipeline


class Backend:
    """
    Represents a Backend.

    Attributes
    ----------
        self._images: Dict[int, Any]
            ...
        self._pre_processing: Dict[str, Any]
            ...
        self._pipeline: DiffusionPipeline
            ...
    """
    _DIFFUSION_PIPELINES = {"stable": StableDiffusionPipeline, "control": ControlDiffusionPipeline}

    def __init__(
        self
    ):
        """ Initializes a Backend. """
        # ----- Attributes ----- #
        # Pre-processing
        self.pre_processing: Dict[str, Any] = {
            "canny": image_processing.Canny,
            "pose": image_processing.Pose,
        }

        # Pipeline
        self.pipeline: DiffusionPipeline = None

    def pre_process_image(
        self,
        image: torch.Tensor,
        pre_process_id: str
    ):
        """
        Pre-processes an image.

        Parameters
        ----------
            image: torch.Tensor
                image to pre-process
            pre_process_id: str
                id of the pre_process to apply
        """
        if isinstance(self.pre_processing[pre_process_id], type):
            self.pre_processing[pre_process_id] = self.pre_processing[pre_process_id]()

        # Pre-process image
        return self.pre_processing[pre_process_id](image=image)

    def process_masks(
        self,
        masks: List[np.ndarray]
    ):
        """
        Pre-processes an image.

        Parameters
        ----------
            masks: List[np.ndarray]
                masks to process
        """
        mask = np.ones_like(masks[0])
        for e in masks:
            mask = e[e > 0]

    def generate_image(
        self,
        pipeline_id: str
    ):
        """
        Generates image

        Parameters
        ----------
            pipeline_id: str
                id of the pipeline to use
        """
        self.pipeline = self._DIFFUSION_PIPELINES[pipeline_id]
        return self.pipeline()
