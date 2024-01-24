# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
from blackboxopt import Evaluation, EvaluationSpecification
from parameterspace import ParameterSpace

from scamlgp.benchmarking.functions.base import Base as FunctionBase

SeedType = Union[
    int, np.random.SeedSequence, np.random.BitGenerator, np.random.Generator
]


class Benchmark(abc.ABC):
    @property
    @abc.abstractmethod
    def target_task(self):
        """
        The target task.

        Returns:
        target_task: Task
        """

    @property
    @abc.abstractmethod
    def meta_tasks(self):
        """
        Dictionary of meta_tasks with key as uid.

        Returns:
        --------
        meta_tasks; dict
        """

    @property
    def function(self) -> FunctionBase:
        """
        Callable function, aka experiment.

        Returns:
        -------
        function: scamlgp.benchmarking.base.Function
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def search_space(self) -> ParameterSpace:
        """Return the benchmark specific search space.

        Returns:
        --------
        parameterspace.ParameterSpace
            Fully described searchspace with valid bounds and a meaningful prior.
        """

    @property
    @abc.abstractmethod
    def output_dimensions(self) -> int:
        """Return the number of output dimensions

        Returns:
        --------
            Number of output dimensions of each individual benchmark evaluation.
        """

    @abc.abstractmethod
    def get_meta_data(
        self, distribution: str, seed: Optional[SeedType] = None
    ) -> Dict[Union[str, int], List[Evaluation]]:
        """Return data for pre-training. Output is noisy data sampled from `num_tasks`
        possible parametrizations of the benchmark (other than defaults). Evaluations at
        `points_per_task` random points within the search space.

        Args:
            distribution: Name of a supported distribution to use for generating data.
            seed: Random seed for reproducibility.

        Returns:
            metadata: dict
                Numerical representation of the meta-data that can be used to metatrain
                a model.

                Keys of the dictionary is task uid and item is a dictionary with
                following entries:
                'X': numpy.array, shape = (n_points_per_task, ),
                'y': numpy.array, shape = (n_points_per_task, ),
                'fidelity': numpy.array, shape = (n_points_per_task, ),
                'descriptors': numpy.array,
                'settings': numpy.array,
                'context': numpy.array
        """

    @staticmethod
    def create_random_task(
        uid, descriptors, settings, context, prng=np.random.RandomState()
    ):
        """Create meta tasks by sampling parameters of the task.

        Parameters:
        ----------
        prng: np.random.RandomState
        """

    @abc.abstractmethod
    def __call__(
        self,
        eval_spec: EvaluationSpecification,
        task_uid: Optional[Union[str, int]] = None,
    ) -> Evaluation:
        """Evaluate the benchmark function at the specified points.

        Parameters:
        -----------
        x: numpy.array, shape = (n_samples, n_features)
            Numerical representation of the points.
        task_uid: int or str
            Unique identifier of the task. Defaults to None

        Returns:
        --------
            Observed value at the query point.
        """


@dataclass(frozen=True)
class Task:
    uid: Union[str, int]
    """Unique identifier of the task"""
    descriptors: Dict[str, Any]
    """Parameters of the underlying function, known to the benchmark but unknown to
    the user (i.e. the meta-learning algorithm). They stay the same for every
    benchmark call. For a real dataset, instead of synthetic functions, there are
    no descriptors.
    """
    settings: Dict[str, Any]
    """Parameters known to both the benchmark as well as the user (i.e. the
    meta-learning algorithm), and the user can choose them (as opposed to context).
    They stay the same for every benchmark call.
    """
    context: Dict[str, Any]
    """Parameters known to both the benchmark as well as the user (i.e. the
    meta-learning algorithm) but the user can't choose them (as opposed to
    settings). They can change on each benchmark call.
    """
