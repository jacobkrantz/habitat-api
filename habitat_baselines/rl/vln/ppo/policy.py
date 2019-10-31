#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

import numpy as np
import torch
import torch.nn as nn
from gym import Space

from habitat import Config
from habitat_baselines.common.utils import CategoricalNet, Flatten
from habitat_baselines.rl.models.instruction_encoder import InstructionEncoder
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.models.simple_cnn import SimpleCNN
from habitat_baselines.rl.ppo.policy import Net, Policy


class VLNBaselinePolicy(Policy):
    def __init__(
        self, observation_space: Space, action_space: Space, vln_config: Config
    ):
        super().__init__(
            VLNBaselineNet(
                observation_space=observation_space, vln_config=vln_config
            ),
            action_space.n,
        )


class VLNBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space: Space, vln_config: Config):
        super().__init__()
        self.vln_config = vln_config

        self.instruction_encoder = InstructionEncoder(
            vln_config.INSTRUCTION_ENCODER
        )

        self.visual_encoder = SimpleCNN(
            observation_space, vln_config.VISUAL_ENCODER.hidden_size
        )

        rnn_input_size = self.instruction_encoder.output_size
        if not self.is_blind:
            rnn_input_size += vln_config.VISUAL_ENCODER.hidden_size

        self.state_encoder = RNNStateEncoder(
            input_size=rnn_input_size,
            hidden_size=vln_config.STATE_ENCODER.hidden_size,
            num_layers=1,
            rnn_type=vln_config.STATE_ENCODER.rnn_type,
        )

        self.train()

    @property
    def output_size(self):
        return self.vln_config.STATE_ENCODER.hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        instruction_embed = self.instruction_encoder(observations)
        x = [instruction_embed]  # size: [batch_size x 512]

        if not self.is_blind:
            perception_embed = self.visual_encoder(
                observations
            )  # size: [batch_size x 512]
            x = [perception_embed] + x

        x = torch.cat(x, dim=1)  # size: [batch_size x 1024]
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states
