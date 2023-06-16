"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: data processing
from PIL import Image

import torch
from torchvision.transforms import Compose, ToTensor, Normalize

# IMPORT: project
from src.backend.text_diffuser import TextDiffuser
from src.backend.text_diffuser.model import get_layout_from_prompt


class Text2ImageDiffuser(TextDiffuser):
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
        super(Text2ImageDiffuser, self).__init__(pipeline_path)

    def _prepare_text_diffuser_inputs(
            self,
            prompt: str,
            num_images: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare the text diffuser input mask.

        Parameters
        ----------
            prompt: str
                prompt to use
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
        processing = Compose([ToTensor(), Normalize([0.5], [0.5])])

        # Creates the text segmentation mask
        text, _ = get_layout_from_prompt(prompt)
        text: torch.Tensor = processing(text).unsqueeze(0).cuda()

        text_segmentation: torch.Tensor = self._segment_image(
            image=text, num_images=num_images
        )

        # Creates the image to fill with the text
        image: torch.Tensor = torch.zeros(num_images, 3, 512, 512).to("cuda")

        image: torch.Tensor = self._vae.encode(image).latent_dist.sample()
        image: torch.Tensor = image * self._vae.config.scaling_factor

        # Creates the features
        feature_mask: torch.Tensor = torch.ones(num_images, 1, 64, 64).to("cuda")

        return text_segmentation, feature_mask, image

    def __call__(
            self,
            prompt: str,
            negative_prompt: str = "",
            num_images: int = 1,
            num_steps: int = 50,
            guidance_scale: float = 7.5,
            latents: torch.Tensor = None,
            seed: int = None
    ) -> Tuple[torch.Tensor, List[Image.Image]]:
        self._scheduler.set_timesteps(num_steps)

        # Creates the randomness controller
        generator = None if seed is None else torch.Generator(device="cpu").manual_seed(seed)
        print(seed, generator)

        # Prepares the starting random noise
        if latents is None:
            latents: torch.Tensor = self._randn(
                b=num_images,
                c=self._vae.config.latent_channels,
                w=512,
                h=512,
                generator=generator
            )
        latents = latents.to("cuda")

        # Prepares the encoded version of the prompt/negative prompt
        encoder_hidden_states, encoder_hidden_states_nocond = self._prepare_prompt(
            prompt=prompt, negative_prompt=negative_prompt, num_images=num_images
        )

        # Prepares the text diffuser inputs
        segmentation_mask, feature_mask, masked_feature = self._prepare_text_diffuser_inputs(
            prompt=prompt, num_images=num_images
        )

        # Generates images
        return latents.cpu(), self._generate_images(
            latents=latents,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_nocond=encoder_hidden_states_nocond,
            segmentation_mask=segmentation_mask,
            feature_mask=feature_mask,
            masked_feature=masked_feature,
            guidance_scale=guidance_scale,
            generator=generator
        )
