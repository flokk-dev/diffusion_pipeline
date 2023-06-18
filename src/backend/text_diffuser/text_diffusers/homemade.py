"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
import os

# IMPORT: data processing
from PIL import Image, ImageDraw, ImageFont

import cv2
import numpy as np

import torch
from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage

# IMPORT: project
import paths
import utils

from src.backend.text_diffuser import TextDiffuser


class MyTextDiffuser(TextDiffuser):
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
        super(MyTextDiffuser, self).__init__(pipeline_path)

    def process_mask(self, mask: np.ndarray):
        pass

    def get_mask(self, image: np.ndarray, prompt: str):
        text = utils.get_key_words(prompt)

        # ----- Make a rectangle from the mask ---- #
        mask = image[:, :, 0]

        # Gets the bounding box of the rectangle
        rows, cols = np.where(mask >= 250)
        if len(rows) == 0 or len(cols) == 0:
            return image

        min_y, max_y = min(rows), max(rows)
        min_x, max_x = min(cols), max(cols)

        # Modifies the mask according to the bounding box
        new_mask = np.full_like(image, fill_value=255)
        new_mask[min_y:max_y, min_x:max_x, :3] = 128

        # ----- Adds text in the mask's bounding box ----- #
        image = Image.fromarray(new_mask)
        drawer = ImageDraw.Draw(image)

        # Adjust font size
        font_path = os.path.join(paths.REQUIREMENTS, "arial.ttf")

        font_size = 500
        while True:
            font = ImageFont.truetype(font_path, font_size)
            text_width, text_height = drawer.textsize(text, font=font)

            if text_width >= (max_x - min_x) or text_height >= (max_y - min_y):
                font_size -= 1
            else:
                break

        # Center the text
        width, height = max_x - min_x, max_y - min_y

        min_x += abs(text_width - width) // 2
        min_y += abs(text_height - height) // 2

        # Draw the text on the mask
        image = Image.fromarray(new_mask)

        drawer = ImageDraw.Draw(image)
        drawer.text(
            (min_x, min_y),
            text,
            font=ImageFont.truetype(font_path, font_size),
            fill=(0, 0, 0)
        )

        return np.asarray(image)

    def _prepare_text_diffuser_inputs(
            self,
            image: np.ndarray,
            mask: np.ndarray,
            prompt: str,
            num_images: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare the text diffuser input mask.

        Parameters
        ----------
            image: np.ndarray
                image to inpaint
            mask: np.ndarray
                mask containing the text to diffuse
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
        mask = self.get_mask(mask, prompt)

        # Converts the image into a tensor
        processing = Compose([ToTensor(), Normalize([0.5], [0.5])])

        image: np.ndarray = cv2.resize(image, (512, 512), interpolation=cv2.INTER_NEAREST)
        image: torch.Tensor = processing(image).unsqueeze(0).cuda()

        # Creates the mask corresponding to the text's bounding box
        text_box: np.ndarray = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        text_box: torch.Tensor = ToTensor()(text_box < 250)[0].float().unsqueeze(0).unsqueeze(0).cuda()

        # Creates the image without the text
        image_without_text: torch.Tensor = image * (1 - text_box)
        image_without_text: torch.Tensor = self._vae.encode(image_without_text).latent_dist.sample()
        image_without_text: torch.Tensor = image_without_text.repeat(num_images, 1, 1, 1) * self._vae.config.scaling_factor

        # Creates the segmentation mask (corresponding to the text)
        text_mask: np.ndarray = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        text_mask: torch.Tensor = ToTensor()(text_mask > 127).float().unsqueeze(0).cuda()

        text_segmentation: torch.Tensor = self._segment_image(image=text_mask, num_images=num_images)

        # Creates the ...
        feature_mask: torch.Tensor = torch.nn.functional.interpolate(text_box, size=(64, 64), mode='nearest')

        return text_segmentation, feature_mask, image_without_text

    def __call__(
            self,
            image: np.ndarray,
            mask: np.ndarray,
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
            image=image, mask=mask, prompt=prompt, num_images=num_images
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
