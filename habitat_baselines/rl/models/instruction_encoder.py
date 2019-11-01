import json

import torch
import torch.nn as nn

from habitat import Config
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder


class InstructionEncoder(nn.Module):
    def __init__(self, config: Config):
        r"""An encoder that uses RNN to encode an instruction. Returns
        the final hidden state after processing the instruction sequence.

        Args:
            config: must have
                vocab_size: number of words in the vocabulary
                embedding_size: The dimension of each embedding vector
                use_pretrained_embeddings:
                embedding_file:
                fine_tune_embeddings:
                dataset_vocab:
                hidden_size: The hidden (output) size
                rnn_type: The RNN cell type.  Must be GRU or LSTM
        """
        super().__init__()

        self.config = config

        if self.config.use_pretrained_embeddings:
            self.embedding_layer = nn.Embedding.from_pretrained(
                embeddings=self._load_embeddings(),
                freeze=not self.config.fine_tune_embeddings,
            )
        else:  # each embedding initialized to sampled Gaussian
            self.embedding_layer = nn.Embedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.embedding_size,
                padding_idx=0,
            )

        rnn = nn.GRU if self.config.rnn_type == "GRU" else nn.LSTM
        self.encoder_rnn = rnn(
            input_size=config.embedding_size, hidden_size=config.hidden_size
        )

    @property
    def output_size(self):
        return self.config.hidden_size

    def _load_embeddings(self):
        """ Loads word embeddings from a pretrained embeddings file.

        PAD: index 0. [0.0, ... 0.0]
        UNK: index 1. mean of all R2R word embeddings: [mean_0, ..., mean_n]
        why UNK is averaged:
            https://groups.google.com/forum/#!searchin/globalvectors/unk|sort:date/globalvectors/9w8ZADXJclA/hRdn4prm-XUJ

        Returns:
            embeddings tensor of size [num_words x embedding_dim]
        """
        print("\nLoading word embeddings...")
        with open(self.config.embedding_file, "r") as f:
            embeddings = torch.tensor(json.load(f))

        print(f"Done loading word embeddings. Dim: {embeddings.size()}.\n")
        return embeddings

    def forward(self, observations):
        """
        Tensor sizes after computation:
            instruction: [batch_size x seq_length]
            lengths: [batch_size]
            hidden_state: [batch_size x hidden_size]
        """
        instruction = observations["instruction"].long()

        # pack_padded_sequence fails if the as_tensor call is not made explicitly
        lengths = (instruction != 0.0).sum(dim=1)
        lengths = torch.as_tensor(lengths, dtype=torch.int64)
        embedded = self.embedding_layer(instruction)

        packed_seq = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )

        _, hidden_state = self.encoder_rnn(packed_seq)

        if self.config.rnn_type == "LSTM":
            hidden_state = hidden_state[0]

        hidden_state = hidden_state.squeeze(0)
        return hidden_state
