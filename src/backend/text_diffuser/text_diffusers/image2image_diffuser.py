"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
import numpy as np

# IMPORT: data processing
from PIL import Image

import torch
from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage

# IMPORT: project
from src.backend.text_diffuser import TextDiffuser, utils


class Image2ImageDiffuser(TextDiffuser):
    """
    Allows to generate images with relevant text using diffusion.

    Attributes
    ----------
        ...: ...
            ...
    """

    def __init__(self, pipeline_path: str):
        """
        Allows to generate images with relevant text using diffusion.

        Parameters
        ----------
            pipeline_path: str
                path to the pretrained pipeline
        """
        # ----- Attributes ----- #
        super(Image2ImageDiffuser, self).__init__(pipeline_path)

    def _prepare_text_diffuser_inputs(
            self,
            image: np.ndarray,
            num_images: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare the text diffuser input mask.

        Parameters
        ----------
            image: np.ndarray
                image containing the text to diffuse
            num_images: int
                number of images to generate

        Returns
        ----------
            torch.Tensor
                ...
            torch.Tensor
                ...
            torch.Tensor
                ...
        """
        # Converts the mask to tensor
        processing = Compose([ToTensor(), Normalize([0.5], [0.5])])
        prompt_mask = processing(image).unsqueeze(0).cuda()

        # Creates the segmentation mask
        with torch.no_grad():
            segmentation_mask = self._load_segmentation_model()(prompt_mask)

        segmentation_mask = segmentation_mask.max(1)[1].squeeze(0)
        segmentation_mask = utils.filter_segmentation_mask(segmentation_mask)
        segmentation_mask = torch.nn.functional.interpolate(
            segmentation_mask.unsqueeze(0).unsqueeze(0).float(), size=(256, 256), mode='nearest'
        )
        segmentation_mask = segmentation_mask.squeeze(1).repeat(num_images, 1, 1).long().to("cuda")

        # Creates the ...
        feature_mask = torch.ones(num_images, 1, 64, 64).to("cuda")

        # Creates the ...
        masked_image = torch.zeros(num_images, 3, 512, 512).to("cuda")

        masked_feature = self._vae.encode(masked_image).latent_dist.sample()
        masked_feature = masked_feature * self._vae.config.scaling_factor

        return segmentation_mask, feature_mask, masked_feature

    def __call__(
            self,
            image: np.ndarray,
            prompt: str,
            num_images: int = 1,
            num_steps: int = 50,
            guidance_scale: float = 7.5
    ) -> list[Image.Image]:
        # Prepares the starting random noise
        latents = self._randn(
            b=num_images,
            c=self._vae.config.latent_channels,
            w=512,
            h=512,
            generator=None
        ).to("cuda")

        # Prepares the encoded version of the prompt/negative prompt
        encoder_hidden_states, encoder_hidden_states_nocond = self._prepare_prompt(
            prompt=prompt, num_images=num_images
        )

        # Prepares the text diffuser inputs
        segmentation_mask, feature_mask, masked_feature = self._prepare_text_diffuser_inputs(
            image=image, num_images=num_images
        )

        # ------------------------------------------------------ #
        generated_images = self._generate_images(
            latents=latents,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_nocond=encoder_hidden_states_nocond,
            segmentation_mask=segmentation_mask,
            feature_mask=feature_mask,
            masked_feature=masked_feature,
            guidance_scale=guidance_scale
        )
        print(type(generated_images))

        # Converts the tensor into PIL images
        return [ToPILImage()(image) for image in generated_images]
