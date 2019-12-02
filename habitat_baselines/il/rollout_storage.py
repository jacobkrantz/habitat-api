#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

import torch


class ILRolloutStorage(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def to(self, device):
        raise NotImplementedError

    @abstractmethod
    def get_forward_pass_data(self, env_steps):
        raise NotImplementedError

    @abstractmethod
    def insert(
        self,
        observations,
        gt_actions,
        masks,
        recurrent_hidden_states,
        action_log_probs,
        episodes_over,
        actions=None,
    ):
        raise NotImplementedError

    @abstractmethod
    def is_full(self):
        raise NotImplementedError

    @abstractmethod
    def after_update(self):
        raise NotImplementedError

    @abstractmethod
    def get_batch(self):
        raise NotImplementedError

    @staticmethod
    def _flatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
        r"""Given a tensor of size (t, n, ..), flatten it to size (t*n, ...).

        Args:
            t: first dimension of tensor.
            n: second dimension of tensor.
            tensor: target tensor to be flattened.

        Returns:
            flattened tensor of size (t*n, ...)
        """
        return tensor.view(t * n, *tensor.size()[2:])


class RolloutStorageFixedBatch(ILRolloutStorage):
    r"""Class for storing rollout information for IL trainers. Collects a set
    number of experience steps from any number of episodes to be used in a
    single batch.
    """

    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
    ):
        self.observations = {}

        for sensor in observation_space.spaces:
            self.observations[sensor] = torch.zeros(
                num_steps + 1,
                num_envs,
                *observation_space.spaces[sensor].shape
            )

        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1,
            num_recurrent_layers,
            num_envs,
            recurrent_hidden_state_size,
        )

        self.action_log_probs = torch.zeros(num_steps, num_envs, 1)
        if action_space.__class__.__name__ == "ActionSpace":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.actions = torch.zeros(num_steps, num_envs, action_shape)
        self.gt_actions = torch.zeros(num_steps, num_envs, action_shape)
        self.prev_gt_actions = torch.zeros(
            num_steps + 1, num_envs, action_shape
        )
        if action_space.__class__.__name__ == "ActionSpace":
            self.actions = self.actions.long()
            self.gt_actions = self.gt_actions.long()
            self.prev_gt_actions = self.prev_gt_actions.long()

        self.masks = torch.ones(num_steps + 1, num_envs, 1)
        self.episodes_over = torch.zeros(num_steps + 1, num_envs, 1)

        self.num_steps = num_steps
        self.num_envs = num_envs
        self.step = 0
        self.just_reset = True  # to know that storage is initially not full

    def to(self, device):
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.gt_actions = self.gt_actions.to(device)
        self.prev_gt_actions = self.prev_gt_actions.to(device)
        self.masks = self.masks.to(device)
        self.episodes_over = self.episodes_over.to(device)

    def get_forward_pass_data(self, env_steps):
        self.just_reset = False

        step_observation = {
            k: v[self.step] for k, v in self.observations.items()
        }

        recurrent_hidden_states_input = (
            self.recurrent_hidden_states[self.step].clone().detach()
        )
        for i in range(self.num_envs):
            if self.episodes_over[self.step, i]:
                recurrent_hidden_states_input[:, :, i] = torch.zeros(
                    recurrent_hidden_states_input[:, :, i].size()
                )
                self.masks[self.step][i] = 1.0
                env_steps[i] = 0

        prev_gt_actions_input = self.prev_gt_actions[self.step]
        masks_input = self.masks[self.step]
        return (
            step_observation,
            recurrent_hidden_states_input,
            prev_gt_actions_input,
            masks_input,
            env_steps,
        )

    def insert(
        self,
        observations,
        gt_actions,
        masks,
        recurrent_hidden_states,
        action_log_probs,
        episodes_over,
        actions=None,
    ):
        r"""TODO: docstring explaining params
        """
        for sensor in observations:
            self.observations[sensor][self.step + 1].copy_(
                observations[sensor]
            )
        self.gt_actions[self.step].copy_(gt_actions)
        self.recurrent_hidden_states[self.step + 1].copy_(
            recurrent_hidden_states
        )
        self.actions[self.step].copy_(actions)
        self.prev_gt_actions[self.step + 1].copy_(gt_actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.masks[self.step + 1].copy_(masks)
        self.episodes_over[self.step + 1].copy_(episodes_over)
        self.step = (self.step + 1) % self.num_steps

    def is_full(self):
        """determine if the batch is done being collected"""
        full = self.step == 0 and not self.just_reset
        self.just_reset = True
        return full

    def after_update(self):
        for sensor in self.observations:
            self.observations[sensor][0].copy_(self.observations[sensor][-1])

        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.prev_gt_actions[0].copy_(self.prev_gt_actions[-1])
        self.masks[0].copy_(self.masks[-1])
        self.episodes_over[0].copy_(self.episodes_over[-1])

    def get_batch(self):
        r"""
        start as:
            self.observations["depth"]: [steps+1 x num_envs x H x W x channels=1]
            self.observations["instruction"]: [steps+1 x num_envs x 153]
            self.observations["rgb"]: [steps+1 x num_envs x 224 x 224 x channels=3]
            self.recurrent_hidden_states: [steps+1 x num_layers x num_envs x hidden_size]
            self.masks: [steps+1 x num_envs x 1]
            self.episodes_over: [steps+1 x num_envs x 1]
            self.prev_gt_actions [steps+1 x num_envs x 1]
            self.gt_actions [steps x num_envs x 1]

        Returns:
            observations_batch: dict of
                rgb: [batch x H x W x channel]
                depth: [batch x H x W x channel]
                instruction: [batch x seq_length]
            recurrent_hidden_states_batch: [num_layers x batch x hidden_size]
            gt_actions_batch: [batch]
            prev_gt_actions_batch: [batch x 1]
            masks_batch: [batch x 1]
            episodes_over_batch: [batch x 1]
            action_log_probs_batch: [batch x 1]
            actions_batch: [batch x 1]
        """
        T = self.num_steps
        N = self.num_envs

        # grab all but the last time step for some variables. These are the
        # same variables to be reset specifically with after_update()
        observations_batch = {k: v[:-1] for k, v in self.observations.items()}
        recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1]
        prev_gt_actions_batch = self.prev_gt_actions[:-1]
        masks_batch = self.masks[:-1]
        episodes_over_batch = self.episodes_over[:-1]

        # flatten all steps and envs to a single batch number
        observations_batch = {
            k: self._flatten_helper(T, N, v)
            for k, v in observations_batch.items()
        }
        recurrent_hidden_states_batch = self._flatten_helper(
            T, N, recurrent_hidden_states_batch.permute(0, 2, 1, 3)
        ).permute(1, 0, 2)

        gt_actions_batch = (
            self._flatten_helper(T, N, self.gt_actions).squeeze().long()
        )
        prev_gt_actions_batch = self._flatten_helper(
            T, N, prev_gt_actions_batch
        )
        masks_batch = self._flatten_helper(T, N, masks_batch)
        episodes_over_batch = self._flatten_helper(T, N, episodes_over_batch)
        action_log_probs_batch = self._flatten_helper(
            T, N, self.action_log_probs
        )
        actions_batch = self._flatten_helper(T, N, self.actions)

        return (
            observations_batch,
            recurrent_hidden_states_batch,
            gt_actions_batch,
            prev_gt_actions_batch,
            masks_batch,
            episodes_over_batch,
            action_log_probs_batch,
            actions_batch,
        )
