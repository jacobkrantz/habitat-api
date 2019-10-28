import torch
import torch.nn as nn

from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder


class InstructionEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        rnn_type: str = "GRU",
    ):
        r"""An encoder that uses RNN to encode an instruction. Returns
        the final hidden state after processing the instruction sequence.

        Args:
            embedding_size: The dimension of each embedding vector
            hidden_size: The hidden (output) size
            rnn_type: The RNN cell type.  Must be GRU or LSTM
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        num_layers = 1 if rnn_type == "GRU" else 2

        # each embedding initialized to sampled Gaussian
        self.embedding_layer = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_size
        )

        self.encoder_rnn = RNNStateEncoder(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            rnn_type=rnn_type,
        )

    def forward(self, observations):
        instruction = observations["instruction"]  # Size: [Batch, length]
        embedded = self.embedding_layer(instruction)
        # for word in embedded:
        #   if word is PAD: # (0)
        #       break
        #   _, rnn_hidden_states = self.encoder_rnn(word, rnn_hidden_states, masks)
        # hidden_states = self.encoder_rnn._unpack_hidden(hidden_states)
        # if self.rnn_type == "GRU":
        #   return hidden_states
        # else:
        #   return hidden_states[0]
        return torch.ones(hidden_size)
