# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import functools
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import scipy.optimize as opt
from blackboxopt import Evaluation, EvaluationSpecification, Objective
from parameterspace import ParameterSpace
from scipy.optimize import OptimizeResult
from scipy.stats.qmc import Sobol

from scamlgp.benchmarking.benchmarks.api import Benchmark, SeedType, Task


def _shgo_minimize(
    eval_func: Callable, search_space: ParameterSpace, objective_name: str = "loss"
) -> OptimizeResult:
    """Minimize the benchmark with simplicial homology global optimization, SHGO

    Original paper: https://doi.org/10.1007/s10898-018-0645-y

    Parameters
    -----------
    benchmark
        The function to be minimized.
    search_space
        Fully described search space with valid bounds and a meaningful prior.

    Returns
    --------
    res
        The optimization result represented as a `OptimizeResult` object.
    """

    bounds_l = []
    bounds_h = []

    bounds = search_space.get_continuous_bounds()
    for bound in bounds:
        bounds_l.append(bound[0])
        bounds_h.append(bound[1])

    # Optimize
    res = opt.shgo(eval_func, bounds=bounds, sampling_method="sobol", n=1024)

    return res


class Base(Benchmark):
    def __init__(
        self,
        descriptors: ParameterSpace,
        settings: ParameterSpace,
        context: ParameterSpace,
        search_space: ParameterSpace,
        target_task: Task,
        meta_tasks: Dict[Union[str, int], Task],
        n_data_per_task: List[int],
        objectives: Optional[List[Objective]] = None,
    ):
        """Create a Base object.

        Parameters
        ----------
        descriptors : Parameters of the underlying function, known to the benchmark but
            unknown to the user (i.e. the meta-learning algorithm). They stay the same
            for every benchmark call. For a real dataset, instead of synthetic
            functions, there are no descriptors.
        settings : Parameters known to both the benchmark as well as the user (i.e. the
            meta-learning algorithm), and the user can choose them (as opposed to
            context). They stay the same for every benchmark call.
        context : Parameters known to both the benchmark as well as the user (i.e. the
            meta-learning algorithm) but the user can't choose them (as opposed to
            settings). They can change on each benchmark call.
        search_space : The arguments of the benchmark that one wants to optimize /
            search over.
        target_task : The configuration of the target task.
        meta_tasks : The configuration for the meta tasks.
        n_data_per_task : Number of observations from each task, i.e. a total of
            `sum(n_data_per_task)` points constitute the meta data.
        objectives: Objectives with name and direction (min/max) returned by the
            benchmark. Defaults to a single "loss" objective to be minimized.
        """
        self._descriptors = descriptors
        self._settings = settings
        self._context = context
        self._search_space = search_space
        self._target_task = target_task
        self._meta_tasks = meta_tasks
        self._n_data_per_task = n_data_per_task
        self._objectives = (
            [Objective("loss", greater_is_better=False)]
            if objectives is None
            else objectives
        )

    @property
    def target_task(self) -> Task:
        return self._target_task

    @property
    def meta_tasks(self) -> Dict[Union[str, int], Task]:
        return self._meta_tasks

    @property
    def search_space(self) -> ParameterSpace:
        return self._search_space

    @property
    def output_dimensions(self) -> int:
        return len(self.objectives)

    @property
    def objectives(self) -> List[Objective]:
        return self._objectives

    @staticmethod
    def create_tasks(
        descriptors,
        settings,
        context,
        num_meta_tasks,
        seed: Optional[SeedType] = None,
    ):
        prng = np.random.default_rng(seed)
        target_task = Base.create_random_task(0, descriptors, settings, context)
        meta_tasks = {
            uid: Base.create_random_task(uid, descriptors, settings, context, prng)
            for uid in range(1, num_meta_tasks + 1)
        }
        return target_task, meta_tasks

    @staticmethod
    def create_random_task(
        uid,
        descriptors: ParameterSpace,
        settings: ParameterSpace,
        context: ParameterSpace,
        seed: Optional[SeedType] = None,
    ):
        """Return Task instance with randomly initialized parameters."""
        prng = np.random.default_rng(seed)
        return Task(
            uid,
            descriptors.sample(rng=prng),
            settings.sample(rng=prng),
            context.sample(rng=prng),
        )

    def __call__(
        self,
        eval_spec: EvaluationSpecification,
        task_uid: Optional[Union[str, int]] = None,
    ) -> Evaluation:
        """Evaluate the benchmark function at the specified points.

        Parameters:
        eval_spec: The settings for all parameters for the evaluation. If present, the
            entries `settings` and `context` will overwrite the task's
            default values.
        task_uid: Unique identifier of the task. Defaults to the target task.

        Returns: Observed value at the query points and the associated costs towards the
            total budget of an optimizer.
        """
        task = self.target_task if task_uid is None else self.meta_tasks[task_uid]

        config = eval_spec.configuration
        # settings is always a dict, but context can be None!
        settings = eval_spec.settings
        context = {} if eval_spec.context is None else eval_spec.context
        # fill in task specific values, if not provided
        for k, v in task.settings.items():
            settings.setdefault(k, v)

        for k, v in task.context.items():
            context.setdefault(k, v)

        objective_values = self.function.__call__(
            **config,
            **task.descriptors,
            **settings,
            **context,
        )
        if not isinstance(objective_values, tuple):
            objective_values = (objective_values,)
        # explicit type hint here because mypy complained
        assert len(self._objectives) == len(objective_values)
        objectives_dict: Dict[str, Optional[float]] = {
            o.name: v for o, v in zip(self._objectives, objective_values)
        }

        return eval_spec.create_evaluation(
            objectives=objectives_dict, user_info={"task_uid": task_uid}
        )

    def get_meta_data(
        self, distribution: str, seed: Optional[SeedType] = None
    ) -> Dict[Union[str, int], List[Evaluation]]:
        """Return data for pre-training. Output is noisy data sampled from each meta
        task. Evaluations at `n_data_per_task` random points within the search space.

        Args:
            distribution: Controls the distribution of the datapoints for the tasks.
                Allowed values are "random" and "sobol".
            seed: Seed for random generator to get the same metadata

        """
        prng = np.random.default_rng(seed)
        sobol = Sobol(d=len(self.search_space), scramble=True, seed=prng)

        meta_data: Dict[Union[str, int], List[Evaluation]] = dict()

        for uid, n_data in zip(self.meta_tasks, self._n_data_per_task):
            if distribution in ["random", "sobol"]:
                meta_data[uid] = []
                for _ in range(n_data):
                    if distribution == "random":
                        config = self.search_space.sample(rng=prng)
                    else:
                        vector = sobol.random().flatten()
                        config = self.search_space.from_numerical(vector)

                    eval_spec = EvaluationSpecification(configuration=config)

                    evaluation = self.__call__(eval_spec, task_uid=uid)
                    meta_data[uid].append(evaluation)
            else:
                raise ValueError(
                    f"Unknown distribution {distribution}, pick 'sobol' or 'random'."
                )

        return meta_data

    def _numpy_wrapper_call(
        self,
        x: np.ndarray,
        context: Dict[str, Any],
        settings: Dict[str, Any],
        task_uid: Optional[Union[str, int]] = None,
        objective_name="loss",
    ):
        """Allow to call with numpy array for the search space to ease optimization with
        `scipy.optimize`.
        """

        eval_spec = EvaluationSpecification(
            configuration=self.search_space.from_numerical(x),
            context=context,
            settings=settings,
        )
        evaluation = self(eval_spec, task_uid=task_uid)
        return evaluation.objectives[objective_name]


def get_minimum(benchmark: Base, task_uid=None):
    task = benchmark.target_task if task_uid is None else benchmark.meta_tasks[task_uid]

    func = functools.partial(
        benchmark._numpy_wrapper_call,
        task_uid=task_uid,
        context=task.context,
        settings=task.settings,
    )
    result = _shgo_minimize(func, benchmark.search_space)
    return result.fun
