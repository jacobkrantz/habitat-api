#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
from collections import defaultdict

import numpy as np

import habitat
from examples.shortest_path_follower_example import SimpleRLEnv
from habitat.core.benchmark import Benchmark
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat_baselines.agents.simple_agents import (
    ForwardOnlyAgent,
    GoalFollower,
    RandomAgent,
    RandomForwardAgent,
)


"""
RandomAgent:
    Takes random actions. If within the goal radius, takes action STOP.
ForwardOnlyAgent:
    Only takes action MOVE_FORWARD. If within the goal radius, takes action STOP.
RandomForwardAgent:
    If within the goal radius, takes action STOP. Else:
    P(MOVE_FORWARD) = 80%
    P(TURN_LEFT) = 10%
    P(TURN_RIGHT) = 10%
GoalFollower:
    Tries to take direct route to the goal and takes action STOP when within
    the goal radius. Turns left or right if |angle_to_goal| > 15 deg.
ShortestPathAgent:
    takes the geodesic shortest path to the goal. If within the goal radius,
    takes action STOP.
"""


def path_follower_benchmark(config, mode="geodesic_path", shortest=False):
    r"""Benchmark for a path following agent.
    Args:
        config: Config
        mode: either "geodesic_path" or "greedy"
        shortest: if True, agent takes shortest path from start to goal. If
            False, agent follows ground truth path from viewpoint to viewpoint
    """
    env = SimpleRLEnv(config=config)
    follower = ShortestPathFollower(
        env.habitat_env.sim, goal_radius=0.5, return_one_hot=False
    )
    follower.mode = mode

    metrics = defaultdict(list)
    for episode in range(len(env.episodes)):
        env.reset()
        episode_id = env.habitat_env.current_episode.episode_id

        steps = 0
        if shortest:
            path = [env.habitat_env.current_episode.goals[0].position]
        else:
            path = env.habitat_env.current_episode.path + [
                env.habitat_env.current_episode.goals[0].position
            ]
        for point in path:
            done = False
            while not done:
                best_action = follower.get_next_action(point)
                if best_action == None:
                    break
                env.step(best_action)
                steps += 1

        obs, reward, done, info = env.step(HabitatSimActions.STOP)
        for k, v in info.items():
            metrics[k].append(v)

    agg_metrics = {}
    for k, v in metrics.items():
        agg_metrics[k] = np.mean(v)

    env.close()
    return agg_metrics


def benchmark(agent, config):
    r"""Benchmark function for habitat.Agent
    """
    env = SimpleRLEnv(config)
    metrics = defaultdict(list)
    for i in range(len(env.episodes)):
        obs = env.reset()
        agent.reset()
        current_episode = env.current_episode
        done = False
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)

        for k, v in info.items():
            metrics[k].append(v)

    agg_metrics = {}
    for k, v in metrics.items():
        agg_metrics[k] = np.mean(v)

    env.close()
    return agg_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-config", type=str, default="configs/tasks/vln_r2r.yaml"
    )
    args = parser.parse_args()
    config_env = habitat.get_config(args.task_config)

    all_metrics = {}
    for agent_name in [
        "RandomAgent",
        "ForwardOnlyAgent",
        "RandomForwardAgent",
        "GoalFollower",
        "PathFollower",
        "ShortestPathFollower",
    ]:
        if agent_name == "PathFollower":
            all_metrics[agent_name] = path_follower_benchmark(config_env)
        elif agent_name == "ShortestPathFollower":
            all_metrics[agent_name] = path_follower_benchmark(
                config_env, shortest=True
            )
        else:
            agent = getattr(sys.modules[__name__], agent_name)(
                config_env.TASK.SUCCESS_DISTANCE,
                config_env.TASK.GOAL_SENSOR_UUID,
            )
            all_metrics[agent_name] = benchmark(agent, config_env)

    for agent_name, metrics in all_metrics.items():
        print(f"Benchmark for agent {agent_name}:")
        for k, v in metrics.items():
            print("{}: {:.3f}".format(k, v))
        print("")


if __name__ == "__main__":
    main()
