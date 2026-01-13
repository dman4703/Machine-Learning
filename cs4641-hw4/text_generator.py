import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextGenerator:
    """Generates text using trained models."""

    def __init__(self, char_indices, indices_char, max_input_len):
        self.char_indices = char_indices
        self.indices_char = indices_char
        self.max_input_len = max_input_len

    def sample(self, preds, temperature=0.5) -> int:
        """Sample next character index based on predictions."""
        preds = np.asarray(preds).astype("float64")
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def generate(self, model_wrapper, seed_text, length=150, temperature=0.5):
        if temperature <= 0:
            print("Temperature must be greater than 0.")
            return

        if len(seed_text) < self.max_input_len:
            seed_text = (" " * (self.max_input_len - len(seed_text))) + seed_text
        else:
            seed_text = seed_text[-self.max_input_len :]

        generated = ""
        print(
            f"-------------------- {model_wrapper.model_name} Model --------------------"
        )
        print("Prompt: " + seed_text)
        print("Model: ", end="")

        model = model_wrapper.model
        model.eval()  # Set the model to evaluation mode

        with torch.no_grad():
            for _ in range(length):
                input_indices = [self.char_indices[char] for char in seed_text]
                x_pred = torch.tensor([input_indices], dtype=torch.long)
                output_log_probs = model(x_pred)
                scaled_logits = output_log_probs / temperature
                probabilities = F.softmax(scaled_logits, dim=1)
                predicted_index = torch.multinomial(probabilities[0], 1).item()
                next_char = self.indices_char[predicted_index]
                generated += next_char
                seed_text = seed_text[1:] + next_char
                print(next_char, end="", flush=True)

        print()  # Final newline
        return generated
