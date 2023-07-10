"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: data processing
import torch

# IMPORT: deep learning
from diffusers import UniPCMultistepScheduler


class Diffuser:
    """
    Allows to generate images using diffusion.

    Attributes
    ----------
        _pipeline: DiffusionPipeline
            diffusion pipeline
    """
    LORA = [
        "flokk/2023-07-04-17-05-55"
    ]

    def __init__(self, pipeline_path: str):
        """
        Allows to generate images using diffusion.

        Parameters
        ----------
            pipeline_path: str
                path to the pretrained pipeline
        """
        # ----- Attributes ----- #
        # Loads the diffusion pipeline
        self._pipeline_path = pipeline_path

        self._pipeline: Any = self._init_pipeline()
        self._pipeline.enable_model_cpu_offload()

        # LoRA
        self._lora_path: str = None

        # Modifies the noise scheduler
        self._pipeline.scheduler = UniPCMultistepScheduler.from_config(
            self._pipeline.scheduler.config
        )

    def is_different(self, pipeline_path: str, lora_path: str) -> bool:
        """
        Checks if the new parameters are different

        Parameters
        ----------
            pipeline_path: str
                new pipeline path
            lora_path: str
                new LoRA path

        Returns
        ----------
            bool
                whether or not the new parameters are different
        """
        if not self._pipeline_path == pipeline_path:
            return True

        if not self._lora_path == lora_path:
            return True

        return False

    def _init_pipeline(self):
        """
        Initializes the diffusion pipeline.

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()

    def _randn(
            self,
            b: int,
            c: int,
            w: int,
            h: int,
            generator: torch.Generator | None
    ) -> torch.Tensor:
        """
        Generates normalized random noise according to the pipeline components.

        Parameters
        ----------
            b: int
                number of random noise
            c: int
                number of channels
            w: int
                width of random noise
            h: int
                height of random noise
            generator: torch.Generator
                randomness controller

        Returns
        ----------
            torch.Tensor
                generated random noise
        """
        # Retrieves the VAE scale factor and the NoiseScheduler standard deviation
        scale_factor = self._pipeline.vae_scale_factor
        sigma = self._pipeline.scheduler.init_noise_sigma

        # Creates the normalized random noise
        latents = torch.randn(size=(b, c, h//scale_factor, w//scale_factor), generator=generator)
        return latents * sigma
