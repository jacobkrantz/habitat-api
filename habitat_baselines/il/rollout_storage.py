#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

import torch

from habitat import logger


class ILRolloutStorage(ABC):
    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        action_space,
        loss_scaling,
        il_algorithm,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
    ):
        super().__init__()
        self.observations = {}
        assert loss_scaling in [
            "TRAJECTORY",
            "INFLECTION",
            "NONE",
        ], "LOSS_SCALING must be one of [TRAJECTORY, INFLECTION, NONE]."
        assert il_algorithm in [
            "TEACHER_FORCING",
            "STUDENT_FORCING",
        ], "IL.ALGORITHM must be either `TEACHER_FORCING` or `STUDENT_FORCING`"

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
        self.prev_actions = torch.zeros(num_steps + 1, num_envs, action_shape)
        if action_space.__class__.__name__ == "ActionSpace":
            self.actions = self.actions.long()
            self.gt_actions = self.gt_actions.long()
            self.prev_actions = self.prev_actions.long()

        self.masks = torch.ones(num_steps + 1, num_envs, 1)
        self.episodes_over = torch.zeros(num_steps + 1, num_envs, 1)

        self.loss_scaling = loss_scaling
        self.il_algorithm = il_algorithm
        self.num_steps = num_steps
        self.num_envs = num_envs

    def to(self, device):
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.gt_actions = self.gt_actions.to(device)
        self.prev_actions = self.prev_actions.to(device)
        self.masks = self.masks.to(device)
        self.episodes_over = self.episodes_over.to(device)

    @abstractmethod
    def get_forward_pass_data(self, env_steps):
        raise NotImplementedError

    @abstractmethod
    def insert(self):
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
        loss_scaling,
        il_algorithm,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
    ):
        super().__init__(
            num_steps,
            num_envs,
            observation_space,
            action_space,
            loss_scaling,
            il_algorithm,
            recurrent_hidden_state_size,
            num_recurrent_layers,
        )
        self.step = 0
        self.just_reset = True  # to know that storage is initially not full

    def get_forward_pass_data(self, env_steps):
        self.just_reset = False

        step_observation = {
            k: v[self.step] for k, v in self.observations.items()
        }

        recurrent_hidden_states_input = self.recurrent_hidden_states[self.step]
        prev_actions_input = self.prev_actions[self.step]
        for i in range(self.num_envs):
            if self.episodes_over[self.step, i]:
                prev_actions_input[i, 0] = 0
                self.masks[self.step][i] = 1.0
                env_steps[i] = 0

        masks_input = self.masks[self.step]
        return (
            step_observation,
            recurrent_hidden_states_input,
            prev_actions_input,
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
        actions,
    ):
        r"""TODO: docstring explaining params
        """

        # if an episode is over, re-initialize next episode
        for i in range(self.num_envs):
            if episodes_over[i]:
                recurrent_hidden_states[:, i] = torch.zeros(
                    recurrent_hidden_states[:, i].size()
                )

        for sensor in observations:
            self.observations[sensor][self.step + 1].copy_(
                observations[sensor]
            )

        self.gt_actions[self.step].copy_(gt_actions)
        self.recurrent_hidden_states[self.step + 1].copy_(
            recurrent_hidden_states
        )
        self.actions[self.step].copy_(actions)
        self.prev_actions[self.step + 1].copy_(gt_actions)
        if self.il_algorithm == "TEACHER_FORCING":
            self.prev_actions[self.step + 1].copy_(gt_actions)
        else:
            self.prev_actions[self.step + 1].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.masks[self.step + 1].copy_(masks)
        self.episodes_over[self.step + 1].copy_(episodes_over)
        self.step = (self.step + 1) % self.num_steps

    def is_full(self):
        """determine if the batch is done being collected"""
        full = self.step == 0 and not self.just_reset
        self.just_reset = True
        return full

    def get_batch(self):
        r"""
        start as:
            self.observations["depth"]: [steps+1 x num_envs x H x W x channels=1]
            self.observations["instruction"]: [steps+1 x num_envs x 153]
            self.observations["rgb"]: [steps+1 x num_envs x 224 x 224 x channels=3]
            self.recurrent_hidden_states: [steps+1 x num_layers x num_envs x hidden_size]
            self.masks: [steps+1 x num_envs x 1]
            self.episodes_over: [steps+1 x num_envs x 1]
            self.prev_actions [steps+1 x num_envs x 1]
            self.gt_actions [steps x num_envs x 1]

        Returns:
            observations_batch: dict of
                rgb: [batch x H x W x channel]
                depth: [batch x H x W x channel]
                instruction: [batch x seq_length]
            recurrent_hidden_states_batch: [num_layers x batch x hidden_size]
            gt_actions_batch: [batch]
            prev_actions_batch: [batch x 1]
            masks_batch: [batch x 1]
            episodes_over_batch: [batch x 1]
            action_log_probs_batch: [batch x 1]
            actions_batch: [batch x 1]
            loss_weights: [batch]
        """
        T = self.num_steps
        N = self.num_envs

        # grab all but the last time step for some variables. These are the
        # same variables to be reset specifically with after_update()
        observations_batch = {k: v[:-1] for k, v in self.observations.items()}
        recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1]
        prev_actions_batch = self.prev_actions[:-1]
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
        prev_actions_batch = self._flatten_helper(T, N, prev_actions_batch)
        masks_batch = self._flatten_helper(T, N, masks_batch)
        episodes_over_batch = self._flatten_helper(T, N, episodes_over_batch)
        action_log_probs_batch = self._flatten_helper(
            T, N, self.action_log_probs
        )
        actions_batch = self._flatten_helper(T, N, self.actions)

        # weights to scale the step-wise gradient
        loss_weights = torch.ones(episodes_over_batch.size(0)).to(
            self.device
        ) / float(episodes_over_batch.size(0))
        return (
            observations_batch,
            recurrent_hidden_states_batch,
            gt_actions_batch,
            prev_actions_batch,
            masks_batch,
            episodes_over_batch,
            action_log_probs_batch,
            actions_batch,
            loss_weights,
        )

    def after_update(self):
        for sensor in self.observations:
            self.observations[sensor][0].copy_(self.observations[sensor][-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.prev_actions[0].copy_(self.prev_actions[-1])
        self.masks[0].copy_(self.masks[-1])
        self.episodes_over[0].copy_(self.episodes_over[-1])


class RolloutStorageEpisodeBased(ILRolloutStorage):
    r"""Collects a set number of steps, then batches only complete episodes.
    """

    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        action_space,
        loss_scaling,
        il_algorithm,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
    ):
        super().__init__(
            num_steps,
            num_envs,
            observation_space,
            action_space,
            loss_scaling,
            il_algorithm,
            recurrent_hidden_state_size,
            num_recurrent_layers,
        )
        self.steps = torch.zeros(num_envs, dtype=torch.int)
        self.pivots = torch.zeros(num_envs, dtype=torch.int) - 1
        self.just_reset = True
        self.device = "cpu"
        self.envs_batch = []

    def to(self, device):
        super().to(device)
        self.device = device

    def get_forward_pass_data(self, env_steps):
        self.just_reset = False
        step_observation = {
            k: torch.stack([v[self.steps[i], i] for i in range(self.num_envs)])
            for k, v in self.observations.items()
        }

        recurrent_hidden_states_input = torch.stack(
            [
                self.recurrent_hidden_states[self.steps[i], :, i]
                for i in range(self.num_envs)
            ],
            dim=1,
        )
        prev_actions_input = torch.stack(
            [self.prev_actions[self.steps[i], i] for i in range(self.num_envs)]
        )
        for i in range(self.num_envs):
            if self.episodes_over[self.steps[i], i]:
                self.masks[self.steps[i], i] = 1.0
                env_steps[i] = 0

        # masks always set to 1.0
        masks_input = self.masks[0]

        return (
            step_observation,
            recurrent_hidden_states_input,
            prev_actions_input,
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
        actions,
    ):
        r"""TODO: docstring explaining params
        """
        for i in range(self.num_envs):
            for sensor in observations:
                self.observations[sensor][self.steps[i] + 1, i].copy_(
                    observations[sensor][i]
                )

            # if the episode is over, reset recurrent state for next episode.
            if episodes_over[i]:
                recurrent_hidden_states[:, i] = torch.zeros(
                    recurrent_hidden_states[:, i].size()
                )

            self.recurrent_hidden_states[self.steps[i] + 1, :, i].copy_(
                recurrent_hidden_states[:, i]
            )
            self.gt_actions[self.steps[i], i].copy_(gt_actions[i])
            self.actions[self.steps[i], i].copy_(actions[i])
            if self.il_algorithm == "TEACHER_FORCING":
                self.prev_actions[self.steps[i] + 1, i].copy_(gt_actions[i])
            else:
                self.prev_actions[self.steps[i] + 1, i].copy_(actions[i])
            self.action_log_probs[self.steps[i], i].copy_(action_log_probs[i])
            self.masks[self.steps[i] + 1, i].copy_(masks[i])
            self.episodes_over[self.steps[i] + 1, i].copy_(episodes_over[i])
        self.steps = (self.steps + 1) % self.num_steps

    def is_full(self):
        """determine if any of the envs are full"""
        full = (self.steps == 0).sum().item() > 0 and not self.just_reset
        self.just_reset = True
        return full

    def _set_pivot(self):
        """Find the step index of the last action for each completed
        episode. This allows us to return a batch of only complete episodes
        for scaling the loss. Incomplete episodes are saved for the next
        update. Sets self.pivots to be a list with dim [num_envs]. Only add
        incomplete envs to batch if it is full.
        """
        self.envs_batch = []
        for i in range(self.num_envs):
            mat = self.episodes_over[:-1, i, 0].nonzero().int()
            if mat.size(0):
                # we need the pivot to be less than the step number
                if self.steps[i] > 0:
                    self.pivots[i] = (
                        ((mat < self.steps[i].to(self.device)).int() * mat)
                        .max()
                        .item()
                    )
                else:
                    self.pivots[i] = mat.max().int().item()
                self.envs_batch.append(i)
            else:
                if self.steps[i] == 0:
                    logger.warn(
                        "WARNING: Batch does not contain a complete episode."
                        + " Falling back to processing the batch as a fixed"
                        + " time horizon."
                    )
                    self.pivots[i] = self.num_steps - 1
                    self.envs_batch.append(i)
                else:
                    self.pivots[i] = -1

    def _scale_by_trajectory(self, episodes_over):
        r"""Compute a weight matrix where each sample is divided by the
        trajectory L1 norm. This way, no episode can be given preferential
        learning treatment by length. The gradient size is [batch x 4].
        Returns:
            loss_weights. Size: [batch]
        """
        episodes_over = episodes_over.squeeze()
        loss_weights = torch.ones(episodes_over.size(0)).to(self.device)
        splits = [-1] + list(
            episodes_over.nonzero().squeeze(dim=1).cpu().numpy()
        )
        if splits[-1] != episodes_over.size(0) - 1:
            splits.append(episodes_over.size(0) - 1)

        for i in range(1, len(splits)):
            div_by = splits[i] - splits[i - 1]
            loss_weights[splits[i - 1] + 1 : splits[i] + 1] /= div_by

        return loss_weights

    def _get_inflection_weights(self, gt_actions, episodes_over):
        r"""Compute the inflection weights [Wijmans et al 2019]. Note that in
        our setting we are batching trajectories together yet still using
        trajectory-based normalization. All normalization of the inflection
        weighting loss is done in the weight coefficients here. Inverse
        inflection frequency was calculated from the training set as (number
        of actions) / (number of inflection points).
        Paper link: https://bit.ly/2suE8oZ
        Args:
            gt_actions: size [batch]
            episodes_over: size [batch x 1]
        Returns:
            inflection_weights: size [batch]
        """
        INFLECTION_FREQ = 3.2

        inflection_weights = torch.ones(gt_actions.size(0), dtype=torch.float)
        inflection_weights[0] = INFLECTION_FREQ
        for i in range(1, len(gt_actions)):
            if gt_actions[i] != gt_actions[i - 1]:
                inflection_weights[i] = INFLECTION_FREQ

        splits = [0] + list(
            episodes_over.squeeze().nonzero().squeeze(dim=1).cpu().numpy() + 1
        )

        if splits[-1] != episodes_over.size(0):
            splits.append(episodes_over.size(0))

        for i in range(1, len(splits)):
            inflection_weights[splits[i - 1] : splits[i]] /= torch.sum(
                inflection_weights[splits[i - 1] : splits[i]]
            )
        return inflection_weights

    def get_batch(self):
        r"""
        start as:
            self.observations["depth"]: [steps+1 x num_envs x H x W x channels=1]
            self.observations["instruction"]: [steps+1 x num_envs x 153]
            self.observations["rgb"]: [steps+1 x num_envs x 224 x 224 x channels=3]
            self.recurrent_hidden_states: [steps+1 x num_layers x num_envs x hidden_size]
            self.prev_actions [steps+1 x num_envs x 1]
            self.gt_actions [steps x num_envs x 1]
            ...

        Returns:
            observations_batch: dict of
                rgb: [batch x H x W x channel]
                depth: [batch x H x W x channel]
                instruction: [batch x seq_length]
            recurrent_hidden_states_batch: [num_layers x batch x hidden_size]
            gt_actions_batch: [batch]
            prev_actions_batch: [batch x 1]
            masks_batch: [batch x 1]
            episodes_over_batch: [batch x 1]
            action_log_probs_batch: [batch x 1]
            actions_batch: [batch x 1]
            loss_weights: [batch]
        """
        self._set_pivot()
        self.envs_batch = list(
            (self.pivots != -1).nonzero().squeeze(dim=1).numpy()
        )

        observations_batch = {
            k: torch.cat([v[: self.pivots[i] + 1, i] for i in self.envs_batch])
            for k, v in self.observations.items()
        }
        recurrent_hidden_states_batch = torch.cat(
            [
                self.recurrent_hidden_states[: self.pivots[i] + 1, :, i]
                for i in self.envs_batch
            ]
        ).permute(1, 0, 2)
        gt_actions_batch = torch.cat(
            [self.gt_actions[: self.pivots[i] + 1, i] for i in self.envs_batch]
        ).squeeze()
        prev_actions_batch = torch.cat(
            [
                self.prev_actions[: self.pivots[i] + 1, i]
                for i in self.envs_batch
            ]
        )
        masks_batch = torch.cat(
            [self.masks[: self.pivots[i] + 1, i] for i in self.envs_batch]
        )
        episodes_over_batch = torch.cat(
            [
                self.episodes_over[: self.pivots[i] + 1, i]
                for i in self.envs_batch
            ]
        )

        action_log_probs_batch = torch.cat(
            [
                self.action_log_probs[: self.pivots[i] + 1, i]
                for i in self.envs_batch
            ]
        )
        actions_batch = torch.cat(
            [self.actions[: self.pivots[i] + 1, i] for i in self.envs_batch]
        )

        # weights to scale the step-wise loss
        if self.loss_scaling == "TRAJECTORY":
            loss_weights = self._scale_by_trajectory(episodes_over_batch)
        elif self.loss_scaling == "INFLECTION":
            loss_weights = self._get_inflection_weights(
                gt_actions_batch, episodes_over_batch
            )
        else:  # "NONE"
            loss_weights = torch.ones(episodes_over_batch.size(0)) / float(
                gt_actions_batch.size(0)
            )
        loss_weights = loss_weights.to(self.device)

        return (
            observations_batch,
            recurrent_hidden_states_batch,
            gt_actions_batch,
            prev_actions_batch,
            masks_batch,
            episodes_over_batch,
            action_log_probs_batch,
            actions_batch,
            loss_weights,
        )

    def after_update(self):
        r"""Shift the data that was not included in the update batch to the
        front. Update steps for each env. There will always be at least one
        environment where steps == 0 (full). All environments may have partial
        episodes that need to be transferred.
        """
        assert self.envs_batch, (
            "self.envs_batch must be set to call `after_update`."
            + "Call `get_batch` or `set_pivot`."
        )

        for i in self.envs_batch:
            # move unfinished episode to front, update step to be just beyond these steps
            if self.steps[i] == 0:
                new_step = self.num_steps - self.pivots[i] - 1
                if new_step != 0:  # a partial ep to transfer
                    self.gt_actions[:new_step, i].copy_(
                        self.gt_actions[self.pivots[i] + 1 :, i]
                    )
                    self.actions[:new_step, i].copy_(
                        self.actions[self.pivots[i] + 1 :, i]
                    )
                    self.action_log_probs[:new_step, i].copy_(
                        self.action_log_probs[self.pivots[i] + 1 :, i]
                    )

                for sensor in self.observations:
                    self.observations[sensor][: new_step + 1, i].copy_(
                        self.observations[sensor][self.pivots[i] + 1 :, i]
                    )
                self.recurrent_hidden_states[: new_step + 1, :, i].copy_(
                    self.recurrent_hidden_states[self.pivots[i] + 1 :, :, i]
                )
                self.prev_actions[: new_step + 1, i].copy_(
                    self.prev_actions[self.pivots[i] + 1 :, i]
                )
                self.masks[: new_step + 1, i].copy_(
                    self.masks[self.pivots[i] + 1 :, i]
                )
                self.episodes_over[: new_step + 1, i].copy_(
                    self.episodes_over[self.pivots[i] + 1 :, i]
                )
            else:
                new_step = self.steps[i] - self.pivots[i] - 1
                if new_step != 0:  # a partial ep to transfer
                    self.gt_actions[:new_step, i].copy_(
                        self.gt_actions[self.pivots[i] + 1 : self.steps[i], i]
                    )
                    self.actions[:new_step, i].copy_(
                        self.actions[self.pivots[i] + 1 : self.steps[i], i]
                    )
                    self.action_log_probs[:new_step, i].copy_(
                        self.action_log_probs[
                            self.pivots[i] + 1 : self.steps[i], i
                        ]
                    )

                for sensor in self.observations:
                    self.observations[sensor][: new_step + 1, i].copy_(
                        self.observations[sensor][
                            self.pivots[i] + 1 : self.steps[i] + 1, i
                        ]
                    )
                self.recurrent_hidden_states[: new_step + 1, :, i].copy_(
                    self.recurrent_hidden_states[
                        self.pivots[i] + 1 : self.steps[i] + 1, :, i
                    ]
                )
                self.prev_actions[: new_step + 1, i].copy_(
                    self.prev_actions[
                        self.pivots[i] + 1 : self.steps[i] + 1, i
                    ]
                )
                self.masks[: new_step + 1, i].copy_(
                    self.masks[self.pivots[i] + 1 : self.steps[i] + 1, i]
                )
                self.episodes_over[: new_step + 1, i].copy_(
                    self.episodes_over[
                        self.pivots[i] + 1 : self.steps[i] + 1, i
                    ]
                )

            self.steps[i] = new_step

        self.pivots = torch.zeros(self.num_envs, dtype=torch.int) - 1
