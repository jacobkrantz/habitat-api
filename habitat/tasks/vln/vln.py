#!/usr/bin/env python3

# Owners/maintainers of the Vision and Language Navigation task:
#   @jacobkrantz: Jacob Krantz
#   @koshyanand: Anand Koshy

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional

import attr
from gym import spaces

from habitat.core.registry import registry
from habitat.core.simulator import Observations, Sensor
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import NavigationEpisode, NavigationTask


@attr.s(auto_attribs=True)
class InstructionData:
    instruction_text: str
    instruction_tokens: Optional[List[str]] = None


@attr.s(auto_attribs=True, kw_only=True)
class VLNEpisode(NavigationEpisode):
    r"""Specification of episode that includes initial position and rotation
    of agent, goal specifications, instruction specifications, reference path,
    and optional shortest paths.

    Args:
        episode_id: id of episode in the dataset
        scene_id: id of scene inside the simulator.
        start_position: numpy ndarray containing 3 entries for (x, y, z).
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation.
        goals: list of goals specifications
        reference_path: List of (x, y, z) positions which gives the reference
            path to the goal that aligns with the instruction.
        instruction: single natural language instruction guide to goal.
        trajectory_id: id of ground truth trajectory path.
    """
    reference_path: List[List[float]] = attr.ib(
        default=None, validator=not_none_validator
    )
    instruction: InstructionData = attr.ib(
        default=None, validator=not_none_validator
    )
    trajectory_id: int = attr.ib(default=None, validator=not_none_validator)


@registry.register_sensor(name="InstructionSensor")
class InstructionSensor(Sensor):
    def __init__(self, **kwargs):
        self.uuid = "instruction"
        self.observation_space = spaces.Discrete(0)

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


@registry.register_measure
class PathLength(Measure):
    r"""Path Length (PL)

    PL = sum(geodesic_distance(agent_prev_position, agent_position)
            over all agent positions.
    """

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position.tolist()
        self._start_end_episode_distance = self._sim.geodesic_distance(
            self._previous_position, episode.goals[0].position
        )
        self._agent_episode_distance = 0.0
        self._metric = None

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()

        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = self._agent_episode_distance

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return "path_length"


@registry.register_measure
class NavigationError(Measure):
    r"""Navigation Error (NE)

    NE = geosdesic_distance(agent_path_end, goal)

    This computes navigation error for every update regardless of whether or
    not the end of the episode has been reached. Thus, this measure is a
    distance to goal measure.
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        super().__init__()

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._metric = None

    def update_metric(self, *args: Any, episode, action, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()
        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )
        self._metric = distance_to_target

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return "navigation_error"


@registry.register_measure
class OracleNavigationError(Measure):
    r"""Oracle Navigation Error (ONE)

    ONE = min(geosdesic_distance(agent_pos, goal))
            over all agent_pos in agent path.

    This computes oracle navigation error for every update regardless of
    whether or not the end of the episode has been reached.
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        super().__init__()

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._metric = float("inf")

    def update_metric(self, *args: Any, episode, action, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()
        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )
        if distance_to_target < self._metric:
            self._metric = distance_to_target

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return "oracle_navigation_error"


@registry.register_measure
class Success(Measure):
    r"""Success Rate (SR)

    SR = I(NE <= goal_radius),
    where NE is Navigation Error.
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        super().__init__()

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._metric = 0

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        self._metric = 0
        current_position = self._sim.get_agent_state().position.tolist()

        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )

        if (
            hasattr(task, "is_stop_called")
            and task.is_stop_called
            and distance_to_target < self._config.SUCCESS_DISTANCE
        ):
            self._metric = 1

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return "success"


@registry.register_measure
class OracleSuccess(Measure):
    r"""Oracle Success Rate (OSR)

    OSR = I(ONE <= goal_radius),
    where ONE is Oracle Navigation Error.
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        super().__init__()

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._metric = 0

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        current_position = self._sim.get_agent_state().position.tolist()
        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )

        if (
            hasattr(task, "is_stop_called")
            and task.is_stop_called
            and distance_to_target < self._config.SUCCESS_DISTANCE
        ):
            self._metric = 1

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return "oracle_success"


@registry.register_measure
class OracleSPL(Measure):
    r"""OracleSPL (Oracle Success weighted by Path Length)

    OracleSPL = max(SPL) over all points in the agent path
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._ep_success = None
        self._sim = sim
        self._config = config
        super().__init__()

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position.tolist()
        self._start_end_episode_distance = episode.info["geodesic_distance"]
        self._agent_episode_distance = 0.0
        self._ep_success = 0
        self._metric = 0.0

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        if self._ep_success:  # shortest path already found
            return

        current_position = self._sim.get_agent_state().position.tolist()

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )
        self._previous_position = current_position

        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )
        if distance_to_target < self._config.SUCCESS_DISTANCE:
            self._ep_success = 1
            self._metric = self._ep_success * (
                self._start_end_episode_distance
                / max(
                    self._start_end_episode_distance,
                    self._agent_episode_distance,
                )
            )

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return "oracle_spl"


@registry.register_measure
class StepsTaken(Measure):
    r"""Counts the number of times update_metric() is called. This is equal to
    the number of times that the agent takes an action. STOP counts as an
    action.
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._metric = 0
        super().__init__()

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._metric = 0

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        self._metric += 1

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return "steps_taken"


@registry.register_task(name="VLN-v0")
class VLNTask(NavigationTask):
    r"""Vision and Language Navigation Task
    Goal: An agent must navigate to a goal location in a 3D environment
        specified by a natural language instruction.
    Metric: Success weighted by Path Length (SPL)
    Usage example:
        examples/vln_reference_path_follower_example.py
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
