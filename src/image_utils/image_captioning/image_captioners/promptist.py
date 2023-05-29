"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: data processing
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# IMPORT: deep learning
from clip_interrogator import Config, Interrogator

# IMPORT: project
from src.image_utils.image_captioning.image_captioner import ImageCaptioner


class Promptist(ImageCaptioner):
    """
    Represents a Promptist.

    Attributes
    ----------
        _model: AutoModelForCausalLM
            model needed to improve the prompts
    """

    def __init__(self):
        """ Initializes a Promptist. """
        super(Promptist, self).__init__()

        # ----- Attributes ----- #
        # Model
        self._model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            "microsoft/Promptist"
        )

        # Tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

    def __call__(
            self,
            prompt: str
    ) -> str:
        """
        Parameters
        ----------
            prompt: str
                prompt to improve

        Returns
        ----------
            str
                improved prompt
        """
        # Tokenizes the prompt
        input_ids = self._tokenizer(prompt.strip() + " Rephrase:", return_tensors="pt").input_ids
        eos_id = self._tokenizer.eos_token_id

        # Improves the prompt
        output = self._model.generate(
            input_ids, do_sample=False, max_new_tokens=75,
            num_beams=8, num_return_sequences=8, eos_token_id=eos_id,
            pad_token_id=eos_id, length_penalty=-1.0
        )

        # Untokenizes the output
        output_texts = self._tokenizer.batch_decode(output, skip_special_tokens=True)
        return output_texts[0].replace(prompt + " Rephrase:", "").strip()
