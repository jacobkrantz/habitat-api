import torch
import torch.nn as nn

from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder


class InstructionEncoder(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, rnn_type: str = "GRU"
    ):
        r"""An encoder that uses RNN to encode an instruction. Returns
        the final hidden state after processing the instruction sequence.

        Args:
            input_size: The input size of the RNN
            hidden_size: The hidden (output) size
            rnn_type: The RNN cell type.  Must be GRU or LSTM
        """
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = 1 if rnn_type == "GRU" else 2
        self.encoder_rnn = RNNStateEncoder(
            input_size, hidden_size, self.num_layers, rnn_type
        )

    def forward(self, observations):
        instruction = observations["instuction"]
        # for word in instruction:
        #   _, rnn_hidden_states = self.encoder_rnn(word, rnn_hidden_states, masks)
        # hidden_states = self.encoder_rnn._unpack_hidden(hidden_states)
        # if self.rnn_type == "GRU":
        #   return hidden_states
        # else:
        #   return hidden_states[0]
        return torch.ones(hidden_size)
