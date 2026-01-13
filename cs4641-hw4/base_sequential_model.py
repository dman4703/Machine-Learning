import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# A migration to PyTorch was made from Tensorflow due to unsupported instructions
# at the hardware level. Unfortunately, it seems whatever Gradescope VM we are utilizing
# runs on the same x86 architecture that doesn't support parts of the backend that torch
# provides. The following line disables it.


class BaseSequentialModel:
    """
    An abstract base class for sequential models (for PyTorch)

    This class provides the foundational structure for building and training
    sequential models, including functionalities for saving model weights and
    tracking loss history.

    Attributes:
        vocab_size (int): The size of the vocabulary, representing the total number of unique tokens.
        max_input_len (int): The maximum length of input sequences.
        model (torch.nn): The pytorch model instance (to be defined in subclasses).
        model_name (str): The name of the model.
        loss_history (dict): A dictionary to store training loss history.
        hyper_params (dict): A dictionary to store hyperparameters for the model training.
    """

    def __init__(self, vocab_size, max_input_len):
        """
        Initializes the BaseSequentialModel with specified vocabulary size and maximum input length.

        Args:
            vocab_size (int): The size of the vocabulary, representing the total number of unique tokens.
            max_input_len (int): The maximum length of input sequences that the model will process.
            hyper_params (dict): A dictionary of hyperparameters for the model
        """

        self.vocab_size = vocab_size
        self.max_input_len = max_input_len
        self.model = None
        self.model_name = ""
        self.loss_history = None
        self.hyper_params = {}

    def save_model_path(self):
        """
        Returns the file path for saving the model weights.

        Returns:
            str: The path to save the model weights, formatted as "rnn_model_weights/{model_name}_weights.keras".
        """

        return f"rnn_model_weights/{self.model_name}_weights.pth"

    def save_losses_path(self):
        """
        Returns the file path for saving the loss history.

        Returns:
            str: The path to save the loss history, formatted as "rnn_model_weights/{model_name}_losses.json".
        """
        return f"rnn_model_weights/{self.model_name}_losses.json"

    def train(self, x, y, train_from_scratch=False):
        """
        Trains the sequential model on the provided data.

        This method attempts to load existing model weights and loss history if they exist.
        If loading fails or if specified, the model will be trained from scratch.

        Args:
            x (array-like): The input data for training.
            y (array-like): The target data for training.
            train_from_scratch (bool): A flag indicating whether to train the model from scratch.
                                        Defaults to False.
        """

        save_model_path = self.save_model_path()
        save_losses_path = self.save_losses_path()

        save_model_path = self.save_model_path()
        save_losses_path = self.save_losses_path()
        os.makedirs("rnn_model_weights", exist_ok=True)

        if train_from_scratch is False and os.path.exists(save_model_path):
            try:
                self.model.load_state_dict(
                    torch.load(save_model_path, map_location="cpu")
                )
                if os.path.exists(save_losses_path):
                    with open(save_losses_path, "r") as f:
                        self.loss_history = json.load(f)
                print(f"Loaded saved {self.model_name} model and weights.")
                return
            except Exception as e:
                print(
                    f"Could not load pre-trained model possibly due to a mismatch in model architecture (Error: {e}). "
                    + "Reverting to training model from scratch."
                )
                self.train(x, y, True)
                return  # Stop and restart with scratch training

        else:
            print(f"Training {self.model_name} model from scratch...")

            batch_size = self.hyper_params.get("batch_size", 64)
            epochs = self.hyper_params.get("epochs", 10)
            lr = self.hyper_params.get("lr", 0.001)

            # Create DataLoader
            x_tensor = torch.tensor(x, dtype=torch.long)
            y_tensor = torch.tensor(y, dtype=torch.long).squeeze()

            dataset = TensorDataset(x_tensor, y_tensor)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Define Loss, Optimizer, and Scheduler
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.RMSprop(self.model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", factor=0.2, patience=1, min_lr=0.001
            )

            self.loss_history = {"losses": []}
            best_loss = float("inf")

            self.model.train()  # Set model to training mode

            for epoch in range(epochs):
                epoch_loss = 0.0
                for batch_x, batch_y in data_loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                # Calculate average loss for the epoch
                avg_epoch_loss = epoch_loss / len(data_loader)
                self.loss_history["losses"].append(avg_epoch_loss)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

                # Scheduler step
                scheduler.step(avg_epoch_loss)

                # Model checkpoint logic (save best model)
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    torch.save(self.model.state_dict(), save_model_path)
                    print(f"  Saved best model to {save_model_path}")

            print(f"Finished training. Best model weights saved to {save_model_path}")

            # Save loss history
            with open(save_losses_path, "w") as f:
                json.dump(self.loss_history, f)
            print(f"Saved {self.model_name} model loss history to {save_losses_path}")

    def plot_loss(self):
        """
        Plots the training loss history.

        If no training history is available, a message is printed prompting the user to train the model first.
        """

        if self.loss_history is None:
            print("No training history available. Train the model first.")
            return

        losses = self.loss_history["losses"]
        plt.figure(figsize=(8, 5))
        plt.plot(losses, "b-")
        plt.title(f"{self.model_name} Loss vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xticks(range(len(losses)))
        plt.tight_layout()
        plt.show()
