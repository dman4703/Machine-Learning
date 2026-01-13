## Q4: Next Character Prediction using Recurrent Neural Networks (RNNs) [7.5% Bonus for All] <span style="color:blue">**[P]**</span> | <span style="color:green">**[W]**</span>

Recurrent Neural Networks are a class of neural networks designed to handle sequential or time-series data, where the order of inputs matters. Unlike feedforward neural networks that treat each input independently, sequential networks maintain memory of previous inputs, making them ideal for tasks involving ordered data like text, time series, or video frames. These networks allow previous outputs to be used as inputs while having hidden states.

Common applications include:
- Text processing (language modeling, translation)
- Machine translation (translating from one language to the other)
- Time series prediction (stock prices, weather forecasting)

In this section, we’ll compare two foundational types of recurrent neural network architectures: Simple Recurrent Neural Networks (Simple RNNs) and Long Short-Term Memory networks (LSTMs). The goal is to train these models to generate text in the style of Macbeth by predicting the next character in a given sequence. This exercise will highlight how each architecture manages sequential dependencies in text generation.

Check out the guide under `utilities/q5_guide` for more details on RNNs.

### Data Preparation

- We'll use Shakespeare's Macbeth from Project Gutenberg
- We vectorize the text by treating every character in our text as an individual unit (e.g., 'macbeth' -> ['m', 'a', 'c'...])
- We use a dictionary to store this mapping: {'a':1, 'b':2, 'c':3, ...}
- This mapping enables bidirectional conversion between characters and integers for model input and output interpretation
- We assign each each character a learnable embedding vector
- Create fixed-sized batches of characters using sliding window approach
- For example, with text "macbeth" (context window=4):
  ```
  Window 1: "macb" → predict "e"
  Window 2: "acbe" → predict "t"
  Window 3: "cbet" → predict "h"
  ```

Our final preprocessed data contains:
- **X**: Input sequences 
    - (shape: [`NUM_SEQUENCES`, `SEQUENCE_LEN`])
    - Contains all character sequences of length SEQUENCE_LEN
- **Y**: Target characters 
    - (shape: [`NUM_SEQUENCES`, `1`])
    - Contains the next character that follows each sequence in X
- **VOCABULARY MAP**: The mapping from all unique characters in the text and their numerical representations
- **VOCAB_SIZE**: Total number of unique characters
- **SEQUENCE_LEN**: Length of input sequences

You can also refer to `preprocess_text_data` located in utilities>utils.py for more details.

### 4.1 Model Architecture [5% Bonus for All] <span style="color:blue">**[P]**</span>

Before diving into the specific architectures, let's understand how data shapes transform through the embedding layer.

Input Sequence Shape Flow:
1. Initial input: `(BATCH_SIZE, SEQUENCE_LEN)`
   - `BATCH_SIZE` sequences containing `SEQUENCE_LEN` integers, where each integer represents a character from our vocabulary
   - Example: If `BATCH_SIZE=32` and `SEQUENCE_LEN=15`, shape is `(32, 15)`

2. Embedding Layer: `(BATCH_SIZE, SEQUENCE_LEN, EMBEDDING_DIM)`
   - Transforms each integer into a vector of size `EMBEDDING_DIM`
   - Example: If `EMBEDDING_DIM=64`:
     - Each character index becomes a vector of 64 numbers
     - Shape expands from `(32, 15)` to `(32, 15, 64)`
     - This means: 32 sequences, each 15 characters long, each character now represented by 64 numbers

#### 4.1.1 Defining the Simple RNN Model

In this part, you need to build a simple recurrent neural network using PyTorch. The architecture of the model is outlined below:

![rnn_architecture](data/images/rnn_architecture.png)

**[EMBEDDING - RNN - ADAPTER - FC]**
> **EMBEDDING**: The Embedding layer maps each integer (representing a character) in the input sequence to a dense vector representation. Each character index becomes a vector of **embedding dimension**. It has an input dimension of **vocab_size** (the total number of unique characters or tokens) and an output dimension defined by **embedding_dim**. This transformation allows the model to capture semantic relationships in the data.
> - Input shape: (batch_size, sequence_length) - A sequence of character indices
> - Output shape: (batch_size, sequence_length, embedding_dim) - Each character transformed into an embedding vector

> **RNN**: This layer processes the sequence data, passing information through time steps to learn temporal patterns. It has **rnn_units** neurons, determining the model's ability to capture dependencies in the sequential data.
> - Input shape: (batch_size, sequence_length, embedding_dim) - Sequence of embedding vectors
> - Output shape: (batch_size, rnn_units) - Final state output

> **RNNOutputAdapter**: Helper function implemented to ensure that the pass to the next layer is of correct dimentionality
> - Input shape (as a tuple): (full_sequence_output of shape (batch_size, sequence_length, rnn_units),
> - final_hidden_state of shape (1, batch_size, rnn_units))
> - Output shape: (batch_size, rnn_units)

> **FC (Dense Layer)**: A fully connected layer that transforms the RNN output to match the number of classes or possible output tokens. It has **vocab_size** neurons, ensuring that each output corresponds to a unique token or class.
> - Input shape: (batch_size, rnn_units) - RNN final state
> - Output shape: (batch_size, vocab_size) - Raw scores for each possible character


You can refer to the following documentation on PyTorch layers for more details:
- [Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- [RNN](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html)
- [Dense](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)

<strong>TODO:</strong> Implement the <strong>define_model</strong> function in <strong>rnn.py</strong>. 

#### 4.1.2 Defining the LSTM Model

In this part, you need to build a long short-term memory (LSTM) network as described below. The architecture of the model is outlined below:

![lstm_architecture](data/images/lstm_architecture.png)

**[EMBEDDING - LSTM - ADAPTER - FC]**
> **EMBEDDING**: The Embedding layer maps each integer in the input sequence to a dense vector representation. It has an input dimension of **vocab_size** (the total number of unique characters or tokens) and an output dimension defined by **embedding_dim**. This transformation allows the model to capture semantic relationships in the data.
> - Input shape: (batch_size, sequence_length) - A sequence of character indices
> - Output shape: (batch_size, sequence_length, embedding_dim) - Each character transformed into an embedding vector

> **LSTM**: This layer processes the sequence data, passing information through time steps to learn temporal patterns. It has **lstm_units** neurons, determining the LSTM ability to capture dependencies in the sequential data.
> - Input shape: (batch_size, sequence_length, embedding_dim) - Sequence of embedding vectors
> - Output shape: (batch_size, lstm_units) - Final state output

> **LSTMOutputAdapter**: Helper function implemented to ensure that the pass to the next layer is of correct dimentionality
> - Input shape (as a tuple): (full_sequence_output of shape (batch_size, sequence_length, rnn_units),
> - final_hidden_state of shape (1, batch_size, rnn_units))
> - Output shape: (batch_size, rnn_units)

> **FC (Dense Layer)**: A fully connected layer that transforms the LSTM output to match the number of classes or possible output tokens. It has **vocab_size** neurons, ensuring that each output corresponds to a unique token or class.
> - Input shape: (batch_size, lstm_units) - LSTM final state
> - Output shape: (batch_size, vocab_size) - Raw scores for each possible character


You can refer to the following documentation on PyTorch layers for more details:
- [Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- [LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [Dense](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)

<strong>TODO:</strong> Implement the <strong>define_model</strong> function in <strong>lstm.py</strong>.
