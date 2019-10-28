#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os
from typing import List, Optional

from habitat.config import Config
from habitat.core.dataset import Dataset
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.datasets.utils import VocabDict
from habitat.tasks.nav.nav import NavigationGoal, ShortestPathPoint
from habitat.tasks.vln.vln import InstructionData, VLNEpisode

CONTENT_SCENES_PATH_FIELD = "content_scenes_path"
DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/mp3d/"

R2R_TRAIN_EPISODES = 10837
R2R_VAL_SEEN_EPISODES = 781
R2R_VAL_UNSEEN_EPISODES = 1839


@registry.register_dataset(name="R2RVLN-v1")
class VLNDatasetV1(Dataset):
    r"""Class inherited from Dataset that loads the MatterPort3D
    Room-to-Room (R2R) dataset for Vision and Language Navigation.
    """

    episodes: List[VLNEpisode]
    instruction_vocab: VocabDict

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(
            config.DATA_PATH.format(split=config.SPLIT)
        ) and os.path.exists(config.SCENES_DIR)

    @staticmethod
    def get_scenes_to_load(config: Config) -> List[str]:
        r"""Return list of scene ids
        """
        assert VLNDatasetV1.check_config_paths_exist(config)
        dataset_dir = os.path.dirname(
            config.DATA_PATH.format(split=config.SPLIT)
        )

        cfg = config.clone()
        cfg.defrost()
        cfg.CONTENT_SCENES = []
        dataset = VLNDatasetV1(cfg)
        return VLNDatasetV1._get_scenes_from_dataset(dataset)

    @staticmethod
    def _get_scenes_from_dataset(dataset):
        scenes = []

        for episode in dataset.episodes:
            scene = episode.scene_id.rsplit("/", 1)[-1].replace(".glb", "")
            if scene not in scenes:
                scenes.append(scene)
        return scenes

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []

        if config is None:
            return

        dataset_filename = config.DATA_PATH.format(split=config.SPLIT)
        with gzip.open(dataset_filename, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:

        deserialized = json.loads(json_str)

        # Done for the serialization test
        if "word_list" in deserialized["instruction_vocab"]:
            self.instruction_vocab = VocabDict(
                word_list=deserialized["instruction_vocab"]["word_list"]
            )
        else:
            self.instruction_vocab = VocabDict(
                word_list=deserialized["instruction_vocab"]
            )

        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]
        for episode in deserialized["episodes"]:
            episode = VLNEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.instruction = InstructionData(**episode.instruction)
            # HACK: the longest instruction_tokens in training is 153. Pad all
            #   instruction_tokens to 200.
            while len(episode.instruction.instruction_tokens) < 200:
                episode.instruction.instruction_tokens.append(
                    self.instruction_vocab.PAD_INDEX
                )

            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = NavigationGoal(**goal)
            self.episodes.append(episode)
