#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""BaselineRegistry is extended from habitat.Registry to provide
registration for trainer and environments, while keeping Registry
in habitat core intact.

Import the baseline registry object using

``from habitat_baselines.common.baseline_registry import baseline_registry``

Various decorators for registry different kind of classes with unique keys

- Register a environment: ``@registry.register_env``
- Register a trainer: ``@registry.register_trainer``
"""

from typing import Optional

from habitat.core.registry import Registry


class BaselineRegistry(Registry):
    @classmethod
    def register_trainer(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a RL training algorithm to registry with key 'name'.

        Args:
            name: Key with which the trainer will be registered.
                If None will use the name of the class.

        """
        from habitat_baselines.common.base_trainer import BaseTrainer

        return cls._register_impl(
            "trainer", to_register, name, assert_type=BaseTrainer
        )

    @classmethod
    def get_trainer(cls, name):
        return cls._get_impl("trainer", name)

    @classmethod
    def register_env(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a environment to registry with key 'name'
            Support subclass of RLEnv or Env.

        Args:
            name: Key with which the env will be registered.
                If None will use the name of the class.

        """
        from habitat import Env, RLEnv

        def wrap_wrap(to_register):
            registered = False
            try:
                reg = cls._register_impl(
                    "env", to_register, name, assert_type=Env
                )
                registered = True
            except AssertionError:
                pass

            try:
                reg = cls._register_impl(
                    "env", to_register, name, assert_type=RLEnv
                )
                registered = True
            except AssertionError:
                pass

            assert (
                registered
            ), f"{to_register} must be a subclass of either {Env} or {RLEnv}"
            return reg

        if to_register is None:
            return wrap_wrap
        return wrap_wrap(to_register)

    @classmethod
    def get_env(cls, name):
        return cls._get_impl("env", name)


baseline_registry = BaselineRegistry()
