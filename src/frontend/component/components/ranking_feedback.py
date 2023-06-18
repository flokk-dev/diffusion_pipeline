"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
import gradio as gr

from PIL import Image
import torch

# IMPORT: project
from src.frontend.component import Component
from src.backend.feedback import RankingFeedback as Algorithm


class RankingFeedback(Component):
    """ Allows to upgrade an image using human feedback. """

    def __init__(self, parent: Any):
        """
        Allows to upgrade an image using human feedback.

        Parameters
        ----------
            parent: Any
                parent of the component
        """
        super(RankingFeedback, self).__init__(parent)

        # ----- Attributes ----- #
        self.parent: Any = parent

        # Algorithm needed to use the feedback
        self.algorithm: Algorithm = None

        # ----- Components ----- #
        self.row_1: gr.Accordion = gr.Accordion(label="Feedback", open=True, visible=False)
        self.row_2: gr.Accordion = gr.Accordion(label="Feedback", open=True, visible=False)
        self.row_3: gr.Accordion = gr.Accordion(label="Feedback", open=True, visible=False)

        # If the algorithm hasn't been instantiated yet
        with self.row_1:
            with gr.Row():
                # Creates the text area allowing to specify the best image
                # Creates the text area allowing to specify the best image
                start_image = gr.Number(
                    label="Index of the best image",
                    interactive=True
                )

                # Creates the text area allowing to specify the number of image of the algorithm
                num_images = gr.Number(
                    label="Number of images to generate",
                    value=4,
                    interactive=True
                )

            # Creates the button allowing to instantiate the algorithm
            button_1: gr.Button = gr.Button("Give feedback", scale=1)
            button_1.click(
                fn=self.instantiate_algorithm,
                inputs=[
                    start_image,
                    num_images,
                    self.parent.image_generation.generated_images
                ],
                outputs=[
                    self.parent.image_generation.generated_images,
                    self.row_1, self.row_2, self.row_3
                ]
            )

        # If the algorithm's step is "gradient_estimation"
        with self.row_2:
            # Creates the text area allowing the user to give its feedback
            ranking = gr.Textbox(
                label="Images ranking",
                placeholder="Please rank the images from best to worst (ex: 6-1-3-4)",
                lines=2,
                interactive=True
            )

            # Creates the button allowing to run a step of the algorithm
            button_2: gr.Button = gr.Button("Give feedback", scale=1)
            button_2.click(
                fn=self.on_click,
                inputs=[
                    ranking,
                    self.parent.image_generation.generated_images
                ],
                outputs=[
                    self.parent.image_generation.generated_images,
                    self.row_1, self.row_2, self.row_3
                ]
            )

        # If the algorithm's step is "line_search"
        with self.row_3:
            # Creates the text area allowing the user to give its feedback
            best_image = gr.Textbox(
                label="Best image",
                placeholder="Please specify the index of the best image",
                lines=2,
                interactive=True
            )

            # Creates the button allowing to run a step of the algorithm
            button_3: gr.Button = gr.Button("Give feedback", scale=1)
            button_3.click(
                fn=self.on_click,
                inputs=[
                    best_image,
                    self.parent.image_generation.generated_images
                ],
                outputs=[
                    self.parent.image_generation.generated_images,
                    self.row_1, self.row_2, self.row_3
                ]
            )

    def instantiate_algorithm(
            self,
            best_image: int,
            num_images: int,
            generated_images: List[Dict[str, Any]]
    ):
        # Verifies the arguments
        best_image = int(best_image) - 1 if best_image >= 0 else 0
        num_images = int(num_images) if num_images > 0 else 1

        # Instantiates the algorithm
        self.algorithm = Algorithm(
            latent=self.parent.latents[best_image],
            image=Image.open(generated_images[best_image]["name"]),
            num_latents=num_images,
            smoothing_factor=0.1,
            lr=2.0
        )

        # Launches an iteration
        latents = self.algorithm.gen_new_latents()
        _, generated_images = self.generate_image(latents)

        # Updates the in memory generated images
        return self.algorithm.adjust_generated_images(generated_images), \
            gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

    def on_click(
            self,
            feedback: str,
            generated_images: List[Dict[str, Any]]
    ):
        # Launches a step of the algorithm
        self.algorithm.ranking_feedback(
            feedback=feedback,
            generated_images=[Image.open(image["name"]) for image in generated_images]
        )

        # Launches an iteration
        latents = self.algorithm.gen_new_latents()
        _, generated_images = self.generate_image(latents)

        # Updates the in memory generated images
        if self.algorithm.step == "gradient_estimation":
            return self.algorithm.adjust_generated_images(generated_images), \
                gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

        elif self.algorithm.step == "line_search":
            return self.algorithm.adjust_generated_images(generated_images), \
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

    def generate_image(
            self,
            latents: torch.Tensor = None
    ) -> Tuple[torch.Tensor, List[Image.Image]]:
        """
        Generates images using the in memory parameters.

        Parameters
        ----------
            latents: torch.Tensor
                latents from which to start the generation
        """
        # Generates images using the diffusion object
        return self.parent.image_generation.diffuser(**{
            **self.parent.args,
            **{"num_images": latents.shape[0], "latents": latents}
        })
