"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: deep learning
from transformers import AutoModelForCausalLM, AutoTokenizer


class Promptist:
    """
    Represents an object allowing to improve prompts

    Attributes
    ----------
        _model: AutoModelForCausalLM
            model allowing to improve prompts
    """

    def __init__(self):
        """ Initializes an object allowing to improve prompts. """
        # ----- Attributes ----- #
        # Model allowing to improve prompts
        self._model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            "microsoft/Promptist"
        )

        # Object allowing to tokenize strings
        self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

    def __call__(self, prompt: str) -> str:
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
        # Transforms the string into a token
        input_ids = self._tokenizer(prompt.strip() + " Rephrase:", return_tensors="pt").input_ids
        eos_id = self._tokenizer.eos_token_id

        # Improves the prompt
        output = self._model.generate(
            input_ids, do_sample=False, max_new_tokens=75,
            num_beams=8, num_return_sequences=8, eos_token_id=eos_id,
            pad_token_id=eos_id, length_penalty=-1.0
        )

        # Transforms the token into a string
        output_texts = self._tokenizer.batch_decode(output, skip_special_tokens=True)
        return output_texts[0].replace(prompt + " Rephrase:", "").strip()
