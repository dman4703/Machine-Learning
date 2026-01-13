import torch
import torch.nn as nn
from base_sequential_model import BaseSequentialModel


class RNNOutputAdapter(nn.Module):
    def forward(self, rnn_output):
        # rnn_output is a tuple: (sequence_of_outputs, final_hidden_state)
        # We only want the final hidden state, which is the second element.
        _, h_n = rnn_output
        # The shape of h_n is (num_layers, batch_size, rnn_units).
        # Since we have one layer, we squeeze the first dimension.
        return h_n.squeeze(0)


class RNN(BaseSequentialModel):
    def __init__(self, vocab_size, max_input_len):
        """
        Initializes the RNN model with specified vocabulary size and maximum input length.

        Args:
            vocab_size (int): The size of the vocabulary, representing the total number of unique characters or tokens.
            max_input_len (int): The maximum length of input sequences that the model will process.

        This constructor also sets the model name to "RNN" and calls the
        initializer of the parent class, BaseSequentialModel, to set up
        additional model attributes.
        """
        super().__init__(vocab_size, max_input_len)
        self.model_name = "RNN"

    def set_hyperparameters(self):
        """
        Sets the hyperparameters for the RNN model.

        This method initializes the hyperparameters used for training the RNN, including:
        - embedding_dim: The dimensionality of the embedding layer.
        - rnn_units: The number of units in the RNN layer.
        - learning_rate: The learning rate for the optimizer.
        - batch_size: The number of samples per gradient update.
        - epochs: The number of epochs to train the model.

        These hyperparameters are stored in the `hp` attribute for easy access during model training.
        """

        self.hp = {
            "embedding_dim": 256,
            "rnn_units": 128,
            "learning_rate": 0.01,
            "batch_size": 128,
            "epochs": 10,
        }

    def define_model(self):
        """
        Defines a sequential RNN model for character-level text generation.

        The model consists of:
        1. An Embedding layer that converts input characters to dense vectors of size embedding_dim
            - Each scalar (character) is transformed into a vector of size embedding_dim
            - Input shape: (batch_size, sequence_length), Output shape: (batch_size, sequence_length, embedding_dim)

        2. A pytorch RNN layer for sequence processing
            - Input shape: (batch_size, sequence_length, embedding_dim)
            - Output shape (as a tuple):
                - (full_sequence_output of shape (batch_size, sequence_length, rnn_units),
                - final_hidden_state of shape (1, batch_size, rnn_units))

        3. RNNOutputAdapter
            - Helper function implemented to ensure that the pass to the next layer is of correct dimentionality
            - Input shape (as a tuple):
                - (full_sequence_output of shape (batch_size, sequence_length, rnn_units),
                - final_hidden_state of shape (1, batch_size, rnn_units))
            - Output shape: (batch_size, rnn_units)

        4. A Dense layer that maps to vocabulary size
            - Input shape: (batch_size, rnn_units), Output shape: (batch_size, vocab_size)

        Hint:
            Use self.hp for getting embedding dimension & RNN units
            Use self.vocab_size for vocabulary size

        Returns:
            None. Sets self.model with the defined nn.Sequential model
        """
        embeddingDim = self.hp["embedding_dim"]
        rnnUnits = self.hp["rnn_units"]
        vocabSize = self.vocab_size

        self.model = nn.Sequential(
            nn.Embedding(vocabSize, embeddingDim),
            nn.RNN(embeddingDim, rnnUnits, batch_first=True),
            RNNOutputAdapter(),
            nn.Linear(rnnUnits, vocabSize)
        )
# end define_model
