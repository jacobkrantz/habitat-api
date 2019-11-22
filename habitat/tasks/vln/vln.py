#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import math
from typing import Any, Dict, List, Optional

import attr
import matplotlib.pyplot as plt
import numpy as np
from dtw import dtw
from fastdtw import fastdtw
from gym import spaces

from habitat.config import Config
from habitat.core.embodied_task import (
    EmbodiedTask,
    Measure,
    SimulatorTaskAction,
)
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    Observations,
    Sensor,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask,
)


@attr.s(auto_attribs=True)
class InstructionData:
    instruction_text: str
    instruction_tokens: Optional[List[str]] = None


@attr.s(auto_attribs=True, kw_only=True)
class VLNEpisode(NavigationEpisode):
    r"""Specification of episode that includes initial position and rotation of
    agent, goal specifications, instruction specifications, and optional shortest paths.

    Args:
        scene_id: id of scene inside the simulator.
        start_position: numpy ndarray containing 3 entries for (x, y, z).
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation.
        instruction: single instruction guide to goal.
        trajectory_id: id of ground truth trajectory path.
        goals: relevant goal object/room.
    """
    path: List[List[float]] = attr.ib(
        default=None, validator=not_none_validator
    )
    instruction: InstructionData = attr.ib(
        default=None, validator=not_none_validator
    )
    trajectory_id: int = attr.ib(default=None, validator=not_none_validator)
    goals: List[NavigationGoal] = attr.ib(
        default=None, validator=not_none_validator
    )


@registry.register_sensor(name="InstructionSensor")
class InstructionSensor(Sensor):
    def __init__(self, **kwargs):
        self.uuid = "instruction"
        # HACK: for when using just tokens and 200 padding
        self.observation_space = spaces.Box(
            low=0, high=5000, shape=(200,), dtype=np.float32
        )

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode: VLNEpisode,
        **kwargs
    ):
        return {
            "text": episode.instruction.instruction_text,
            "tokens": episode.instruction.instruction_tokens,
            "trajectory_id": episode.trajectory_id,
        }

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)


@registry.register_task(name="VLN-v0")
class VLNTask(NavigationTask):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


@registry.register_measure
class NDTW(Measure):
    r"""NDTW (Normalized Dynamic Time Warping) & SDTW (Success Weighted be NDTW)

    ref: Effective and General Evaluation for Instruction
        Conditioned Navigation using Dynamic Time
        Warping - Magalhaes et. al
    https://arxiv.org/pdf/1907.05446.pdf
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self.locations = []
        self._sim = sim
        self._config = config
        gt_path = config.GT_PATH.format(split=config.SPLIT)
        with gzip.open(gt_path, "rt") as f:
            self.gt_json = json.load(f)
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "ndtw"

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self.locations.clear()
        self._metric = None

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        current_position = self._sim.get_agent_state().position.tolist()
        if len(self.locations) == 0:
            self.locations.append(current_position)
        else:
            if current_position != self.locations[-1]:
                self.locations.append(current_position)
            else:
                return
        gt_locations = self.gt_json[str(episode.episode_id)]["locations"]

        if self._config.FDTW:
            distance, path = fastdtw(
                self.locations, gt_locations, dist=self._euclidean_distance
            )
        else:
            distance, cost_matrix, acc_cost_matrix, path = dtw(
                self.locations, gt_locations, dist=self._euclidean_distance
            )

        nDTW = math.exp(
            -distance / (len(gt_locations) * self._config.SUCCESS_DISTANCE)
        )

        metrics = []
        if "NDTW" in self._config.METRICS:
            metrics.append(nDTW)

        if "SDTW" in self._config.METRICS:
            ep_success = 0

            distance_to_target = self._sim.geodesic_distance(
                current_position, episode.goals[0].position
            )
            if (
                hasattr(task, "is_stop_called")
                and task.is_stop_called
                and distance_to_target < self._config.SUCCESS_DISTANCE
            ):
                ep_success = 1

            metrics.append(ep_success * nDTW)

        self._metric = metrics
