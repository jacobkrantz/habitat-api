#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os
import time
from collections import deque
import time
from typing import Dict, List

import numpy as np
import torch
from gym import spaces
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations.utils import (
    append_text_to_image,
    observations_to_image,
)
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_utils import construct_envs_auto_reset_false
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.il.policy import ILAgent, VLNILBaselinePolicy
from habitat_baselines.il.rollout_storage import (
    RolloutStorageEpisodeBased,
    RolloutStorageFixedBatch,
)
from habitat_baselines.rl.vln.ppo.utils import transform_observations

from habitat_baselines.common.utils import (  # linear_decay,
    generate_video,
    batch_obs,
)


@baseline_registry.register_trainer(name="imitation_vln")
class ILVLN_Trainer(BaseRLTrainer):
    r"""Trainer class for Imitation Learning algorithms.
    Inherits from BaseRLTrainer for functions:
        - eval()
        - _setup_eval_config()
        - and pause_envs()
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.agent = None
        if config is not None:
            logger.info(f"config: {config}")
        with gzip.open(
            config.IL.GT_PATH.format(split=config.TASK_CONFIG.DATASET.SPLIT),
            "rt",
        ) as f:
            self.gt_data = json.load(f)

    def save_checkpoint(self, file_name: str) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    def _setup_agent(self, config):
        r"""Sets up an agent for imitation learning.

        Args:
            config: VLN config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        # Add TORCH_GPU_ID to VLN config for a ResNet layer
        config.defrost()
        config.TORCH_GPU_ID = self.config.TORCH_GPU_ID
        config.freeze()

        self.policy = VLNILBaselinePolicy(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            vln_config=config,
        )
        self.policy.to(self.device)

        self.agent = ILAgent(
            net=self.policy, lr=self.config.IL.lr, eps=self.config.IL.eps
        )

    def get_gt_actions(self, observations, episodes, steps):
        r""" For each env, get the ground truth action an agent should take.
        Args:
            observations: dict with keys 'instruciton', 'rgb', and 'depth'.
                Currently not used.
            episodes: list of current episodes each with attribute episode_id.
            steps: [num_envs] specifies what step to query the gt_actions for.
        Returns:
            gt_actions: [num_envs x 1]
        """
        actions = []
        for i in range(self.envs.num_envs):
            episode_id = str(episodes[i].episode_id)
            step = steps[i]
            if step > 498:
                logger.warning(f"{step} steps taken in episode {episode_id}.")
            try:
                action = self.gt_data[episode_id]["actions"][step]
            except Exception as e:
                logger.error("Error trying to load ground truth action.")
                logger.error(f"Episode: {episode_id}.")
                raise e
            actions.append(action)
        return torch.tensor(actions).unsqueeze(dim=1).to(self.device)

    def initialize_rollouts(self):
        observations = self.envs.reset()
        observations = transform_observations(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        # batch["rgb"]: torch.Size([1, 224, 224, 3])
        # batch["depth"]: torch.Size([1, 224, 224, 1])
        # batch["instruction"]: torch.Size([1, 153])
        batch = batch_obs(observations)

        # set the observation space for instructions according to config
        for i in range(len(self.envs.observation_spaces)):
            self.envs.observation_spaces[i].spaces["instruction"] = spaces.Box(
                low=0,
                high=self.config.RL.VLN.INSTRUCTION_ENCODER.vocab_size,
                shape=(self.config.RL.VLN.INSTRUCTION_ENCODER.max_length,),
                dtype=np.float32,
            )

        rollouts = globals()[self.config.IL.ROLLOUT_CLASS](
            self.config.IL.BATCH_SIZE,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            self.config.IL.LOSS_SCALING,
            self.config.RL.VLN.STATE_ENCODER.hidden_size,
        )
        rollouts.to(self.device)

        # Initialize rollout observations: [0] for the 0th step in batch
        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])
        return rollouts

    def _collect_rollout_step(self, rollouts, env_steps):
        r"""
        Using the previously collected envirnment rollout, perform a batch
        of forward passes with the agent. Save predictions to rollout storage
        so they can be backpropogated as one minibatch. Collect N steps of GT
        actions to step the environments.

        Resets env_steps[i] if episode switches in env[i]
        Args:
            env_steps: what the step count is at for each episode.
        """
        t_rollout = time.time()
        t_env = 0.0
        with torch.no_grad():
            (
                step_observation,
                recurrent_hidden_states_input,
                prev_gt_actions_input,
                masks_input,
                env_steps,
            ) = rollouts.get_forward_pass_data(env_steps)

            (
                agent_actions,
                agent_actions_log_probs,
                recurrent_hidden_states,
            ) = self.policy.act(
                step_observation,
                recurrent_hidden_states_input,
                prev_gt_actions_input,
                masks_input,
                deterministic=True,
            )

        gt_actions = self.get_gt_actions(
            step_observation, self.envs.current_episodes(), env_steps
        )

        t_env_step = time.time()
        gt_observations = self.envs.step([a.item() for a in gt_actions])
        env_steps += 1
        t_env += time.time() - t_env_step

        masks = torch.tensor(
            [
                [0.0] if a == HabitatSimActions.STOP else [1.0]
                for a in gt_actions
            ],
            dtype=torch.float,
        )
        episodes_over = torch.tensor(
            [
                int(self.envs.call_at(i, "get_episode_over"))
                for i in range(self.envs.num_envs)
            ],
            dtype=torch.int,
        ).unsqueeze(dim=1)

        # reset envs and observations if necessary
        t_env_step = time.time()
        for i in range(self.envs.num_envs):
            if episodes_over[i]:
                gt_observations[i] = self.envs.reset_at(i)[0]
        t_env += time.time() - t_env_step

        gt_observation_batch = batch_obs(
            transform_observations(
                gt_observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            )
        )

        rollouts.insert(
            gt_observation_batch,
            gt_actions,
            masks,
            recurrent_hidden_states,
            agent_actions_log_probs,
            episodes_over,
            actions=agent_actions,
        )

        # Compute monitoring stats
        t_rollout = time.time() - t_rollout - t_env
        delta_episodes = episodes_over.sum().item()
        delta_steps = self.envs.num_envs
        matching_actions = (agent_actions == gt_actions).sum()

        return (
            delta_episodes,
            delta_steps,
            matching_actions,
            env_steps,
            t_env,
            t_rollout,
        )

    def _update_agent(self, rollouts):
        t_update_model = time.time()
        loss, dist_entropy = self.agent.update(rollouts)
        rollouts.after_update()
        t_update_model = time.time() - t_update_model
        return loss, dist_entropy, t_update_model

    def train(self) -> None:
        r"""Main method for training with Teacher Forcing.

        Returns:
            None
        """
        self.envs = construct_envs_auto_reset_false(
            self.config, get_env_class(self.config.ENV_NAME)
        )
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_agent(self.config.RL.VLN)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        rollouts = self.initialize_rollouts()
        env_steps = torch.zeros(self.envs.num_envs, dtype=torch.int)
        window_correct_steps = deque(
            maxlen=self.config.IL.window_accuracy_size
        )

        count_steps = 0
        count_checkpoints = 0
        episodes_seen = 0
        t_update_model = 0.0
        t_env = 0.0
        t_rollout = 0.0

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for update in range(self.config.NUM_UPDATES):
                while not rollouts.is_full():
                    (
                        delta_episodes,
                        delta_steps,
                        matching_actions,
                        env_steps,
                        t_env_single,
                        t_rollout_single,
                    ) = self._collect_rollout_step(rollouts, env_steps)

                    count_steps += delta_steps
                    episodes_seen += delta_episodes
                    window_correct_steps.append(matching_actions.clone())
                    t_env += t_env_single
                    t_rollout += t_rollout_single

                (
                    loss,
                    dist_entropy,
                    t_update_model_single,
                ) = self._update_agent(rollouts)

                t_update_model += t_update_model_single

                # log stats
                step_acc = round(
                    torch.stack(list(window_correct_steps), dim=0)
                    .sum()
                    .sum()
                    .item()
                    / (len(window_correct_steps) * self.envs.num_envs),
                    5,
                )

                writer.add_scalar("loss", loss.item(), count_steps)
                writer.add_scalar("step_accuracy", step_acc, count_steps)
                writer.add_scalar("dist_entropy", dist_entropy, count_steps)

                if update > 0 and update % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        f"update: {update} steps: {count_steps} episodes: {episodes_seen}\t"
                    )
                    logger.info(
                        f"Window size {len(window_correct_steps)} step action accuracy: {step_acc}"
                    )
                    logger.info(
                        f"Compute agent and GT action time (s): {round(t_rollout)}"
                    )
                    logger.info(f"Simulator time (s): {round(t_env)}")
                    logger.info(f"Update time (s): {round(t_update_model)}")

                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(f"ckpt.{count_checkpoints}.pth")
                    count_checkpoints += 1

        self.envs.close()

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        test_recurrent_hidden_states,
        not_done_masks,
        prev_actions,
        batch,
    ):
        # pausing self.envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            test_recurrent_hidden_states = test_recurrent_hidden_states[
                :, state_index
            ]
            not_done_masks = not_done_masks[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

        return (
            envs,
            test_recurrent_hidden_states,
            not_done_masks,
            prev_actions,
            batch,
        )

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint. Assumes episode IDs are unique.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        logger.info(f"checkpoint_path: {checkpoint_path}")
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        # setup agent
        self.envs = construct_envs_auto_reset_false(
            config, get_env_class(config.ENV_NAME)
        )
        self.device = (
            torch.device("cuda", config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self._setup_agent(config.RL.VLN)
        self.agent.load_state_dict(ckpt_dict["state_dict"])

        observations = self.envs.reset()
        observations = transform_observations(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(observations, self.device)

        eval_recurrent_hidden_states = torch.zeros(
            1,  # num_recurrent_layers
            self.config.NUM_PROCESSES,
            self.config.RL.VLN.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )
        stats_episodes = {}  # dict of dicts that stores stats per episode

        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            rgb_frames = [[] for _ in range(self.config.NUM_PROCESSES)]

        # loop for each step to cover all test episodes
        while (
            self.envs.num_envs
            > 0
            # and len(stats_episodes) < self.config.TEST_EPISODE_COUNT
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (
                    actions,
                    _,
                    eval_recurrent_hidden_states,
                ) = self.agent.net.act(
                    batch,
                    eval_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=True,
                )
                prev_actions.copy_(actions)

            observations = self.envs.step([a.item() for a in actions])
            episodes_over = torch.tensor(
                [
                    int(self.envs.call_at(i, "get_episode_over"))
                    for i in range(self.envs.num_envs)
                ],
                dtype=torch.int,
            )
            # thanks to these masks, we don't need to reset the RNN state
            not_done_masks = episodes_over.clone().float().to(self.device)
            not_done_masks[episodes_over == 1] = 0
            not_done_masks[episodes_over == 0] = 1

            # reset envs and observations if necessary
            for i in range(self.envs.num_envs):
                if len(self.config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(
                        observations[i], self.envs.call_at(i, "get_metrics")
                    )
                    frame = append_text_to_image(
                        frame, current_episodes[i].instruction.instruction_text
                    )
                    rgb_frames[i].append(frame)

                if not episodes_over[i]:
                    continue

                stats_episodes[
                    current_episodes[i].episode_id
                ] = self.envs.call_at(i, "get_metrics")
                observations[i] = self.envs.reset_at(i)[0]
                prev_actions[i] = torch.zeros(1, dtype=torch.long)

                if len(self.config.VIDEO_OPTION) > 0:
                    generate_video(
                        video_option=self.config.VIDEO_OPTION,
                        video_dir=self.config.VIDEO_DIR,
                        images=rgb_frames[i],
                        episode_id=current_episodes[i].episode_id,
                        checkpoint_idx=checkpoint_index,
                        metric_name="SPL",
                        metric_value=round(
                            stats_episodes[current_episodes[i].episode_id][
                                "spl"
                            ],
                            6,
                        ),
                        tb_writer=writer,
                    )

                    del stats_episodes[current_episodes[i].episode_id][
                        "top_down_map"
                    ]
                    del stats_episodes[current_episodes[i].episode_id][
                        "collisions"
                    ]
                    rgb_frames[i] = []

            observations = transform_observations(
                observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            )
            batch = batch_obs(observations, self.device)

            envs_to_pause = []
            next_episodes = self.envs.current_episodes()

            for i in range(self.envs.num_envs):
                if next_episodes[i].episode_id in stats_episodes:
                    envs_to_pause.append(i)

            (
                self.envs,
                eval_recurrent_hidden_states,
                not_done_masks,
                prev_actions,
                batch,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                eval_recurrent_hidden_states,
                not_done_masks,
                prev_actions,
                batch,
            )

        self.envs.close()

        split = config.TASK_CONFIG.DATASET.SPLIT
        with open(f"stats_episodes_{split}.json", "w") as f:
            json.dump(stats_episodes, f, indent=4)

        time.sleep(5)
        aggregated_stats = {}
        num_episodes = len(stats_episodes)
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum([v[stat_key] for v in stats_episodes.values()])
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.6f}")
