#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

import numpy as np
import torch
import torch.nn as nn

from habitat_baselines.common.utils import CategoricalNet, Flatten
from habitat_baselines.rl.models.instruction_encoder import InstructionEncoder
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.models.simple_cnn import SimpleCNN
from habitat_baselines.rl.ppo.policy import Net, Policy


class VLNBaselinePolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        instruction_sensor_uuid,
        vocab_size,
        hidden_size=512,
    ):
        super().__init__(
            VLNBaselineNet(
                observation_space=observation_space,
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                instruction_sensor_uuid=instruction_sensor_uuid,
            ),
            action_space.n,
        )


class VLNBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space,
        vocab_size,
        hidden_size,
        instruction_sensor_uuid,
    ):
        super().__init__()

        self.instruction_sensor_uuid = instruction_sensor_uuid
        self._instruction_embedding_size = 200  # HACK
        self._hidden_size = hidden_size

        self.instruction_encoder = InstructionEncoder(
            vocab_size=vocab_size,
            embedding_size=self._instruction_embedding_size,
            hidden_size=self._hidden_size,
        )

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self._hidden_size)
            + self._instruction_embedding_size,
            self._hidden_size,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        instruction_embed = self.instruction_encoder(observations)
        x = [instruction_embed]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            x = [perception_embed] + x

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states
