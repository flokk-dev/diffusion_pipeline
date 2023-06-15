"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

import os
from tqdm import tqdm

# IMPORT: data processing
from PIL import Image

import torch
from torchvision import transforms

# IMPORT: project
import paths

from src.backend.text_diffuser import TextDiffuser, utils
from src.backend.text_diffuser.model import UNet, get_layout_from_prompt


class ImageInpaintDiffuser(TextDiffuser):
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
        super(ImageInpaintDiffuser, self).__init__(pipeline_path)

    def __call__(
            self,
            image: torch.Tensor,
            mask: torch.Tensor,
            prompt: str,
            num_images: int = 1,
            num_steps: int = 50,
            guidance_scale: float = 7.5
    ) -> list[Image.Image]:
        #
        noise = torch.randn((num_images, 4, 64, 64)).to("cuda")
        input = noise

        captions = [prompt] * num_images
        captions_nocond = [""] * num_images
        print(f"captions_nocond: {prompt}.")

        # encode text prompts
        inputs = self._tokenizer(
            captions, max_length=self._tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        ).input_ids
        encoder_hidden_states = self._text_encoder(inputs)[0].cuda()
        print(f"encoder_hidden_states: {encoder_hidden_states.shape}.")

        inputs_nocond = self._tokenizer(
            captions_nocond, max_length=self._tokenizer.model_max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        ).input_ids
        encoder_hidden_states_nocond = self._text_encoder(inputs_nocond)[0].cuda()
        print(f"encoder_hidden_states_nocond: {encoder_hidden_states_nocond.shape}.")

        # load character-level segmenter
        segmenter = UNet(3, 96, True).cuda()
        segmenter = torch.nn.DataParallel(segmenter)
        segmenter.load_state_dict(
            torch.load(os.path.join(paths.TEXT_DIFFUSER, "text_segmenter.pth"))
        )
        segmenter.eval()

        # ------------------------------------------------------ #
        text_mask_tensor = transforms.ToTensor()(mask).unsqueeze(0).cuda().sub_(0.5).div_(0.5)
        with torch.no_grad():
            segmentation_mask = segmenter(text_mask_tensor)

        segmentation_mask = segmentation_mask.max(1)[1].squeeze(0)
        segmentation_mask = utils.filter_segmentation_mask(segmentation_mask)
        segmentation_mask = torch.nn.functional.interpolate(
            segmentation_mask.unsqueeze(0).unsqueeze(0).float(), size=(256, 256), mode='nearest')

        image_mask = torch.ones_like(text_mask_tensor) - text_mask_tensor
        image_mask = torch.from_numpy(image_mask).cuda().unsqueeze(0).unsqueeze(0)

        image_tensor = transforms.ToTensor()(image).unsqueeze(0).cuda().sub_(0.5).div_(0.5)
        masked_image = image_tensor * (1 - image_mask)
        masked_feature = self._vae.encode(masked_image).latent_dist.sample().repeat(num_images,
                                                                                    1, 1, 1)
        masked_feature = masked_feature * self._vae.config.scaling_factor

        image_mask = torch.nn.functional.interpolate(image_mask, size=(256, 256),
                                                     mode='nearest').repeat(num_images, 1, 1, 1)
        segmentation_mask = segmentation_mask * image_mask
        feature_mask = torch.nn.functional.interpolate(image_mask, size=(64, 64), mode='nearest')
        print(f"feature_mask: {feature_mask.shape}.")
        print(f"segmentation_mask: {segmentation_mask.shape}.")
        print(f"masked_feature: {masked_feature.shape}.")

        # ------------------------------------------------------ #

        intermediate_images = []
        for t in tqdm(self._scheduler.timesteps):
            with torch.no_grad():
                noise_pred_cond = self._unet(sample=input, timestep=t,
                                       encoder_hidden_states=encoder_hidden_states,
                                       segmentation_mask=segmentation_mask,
                                       feature_mask=feature_mask,
                                       masked_feature=masked_feature).sample  # b, 4, 64, 64
                noise_pred_uncond = self._unet(sample=input, timestep=t,
                                         encoder_hidden_states=encoder_hidden_states_nocond,
                                         segmentation_mask=segmentation_mask,
                                         feature_mask=feature_mask,
                                         masked_feature=masked_feature).sample  # b, 4, 64, 64
                noisy_residual = noise_pred_uncond + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                )  # b, 4, 64, 64
                prev_noisy_sample = self._scheduler.step(noisy_residual, t, input).prev_sample
                input = prev_noisy_sample
                intermediate_images.append(prev_noisy_sample)

        # decode and visualization
        input = 1 / self._vae.config.scaling_factor * input
        sample_images = self._vae.decode(input.float(), return_dict=False)[0]  # (b, 3, 512, 512)

        # save pred_img
        pred_image_list = []
        for image in sample_images.float():
            image = (image / 2 + 0.5).clamp(0, 1).unsqueeze(0)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = Image.fromarray((image * 255).round().astype("uint8")).convert('RGB')
            pred_image_list.append(image)

        return pred_image_list
