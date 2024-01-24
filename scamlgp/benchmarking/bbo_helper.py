# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import inspect
from typing import Any, Dict, List, Tuple, Type, Union

from blackboxopt import Evaluation, Objective
from blackboxopt.base import MultiObjectiveOptimizer, SingleObjectiveOptimizer
from blackboxopt.optimization_loops import sequential

from scamlgp.benchmarking.benchmarks.base import Base as BenchmarkBase


def _prep_objective_and_optimizer_kwargs(
    benchmark: BenchmarkBase,
) -> Tuple[Objective, Dict[str, Any]]:
    """Optimizer specific additions to the initialization, as well as (modeled-)
    objective preparation.
    """
    objective: Objective

    objective = Objective("loss", greater_is_better=False)
    if hasattr(benchmark, "objectives"):
        objective = benchmark.objectives[0]

    return objective


def _init_optimizer(
    benchmark: BenchmarkBase,
    optimizer_cls: Type[Union[SingleObjectiveOptimizer, MultiObjectiveOptimizer]],
    objective: Objective,
    optimizer_kwargs: Dict[str, Any],
) -> Tuple[List[Objective], Union[SingleObjectiveOptimizer, MultiObjectiveOptimizer]]:
    """Takes care of initializing single and multi objective optimizer variants and
    returns the used objectives alongside the optimizer instance.
    """
    if issubclass(optimizer_cls, SingleObjectiveOptimizer):
        single_objective_optimizer = optimizer_cls(
            search_space=benchmark.search_space,
            objective=objective,
            **optimizer_kwargs,
        )
        return [objective], single_objective_optimizer

    if issubclass(optimizer_cls, MultiObjectiveOptimizer):
        objectives = (
            benchmark.objectives if hasattr(benchmark, "objectives") else [objective]
        )
        multi_objective_optimizer = optimizer_cls(
            search_space=benchmark.search_space,
            objectives=objectives,
            **optimizer_kwargs,
        )
        return objectives, multi_objective_optimizer

    raise ValueError(f"Unsupported subclass of optimizer {optimizer_cls.__base__}")


def run_with_bbo(
    benchmark: BenchmarkBase,
    optimizer_cls: Union[Type[SingleObjectiveOptimizer], Type[MultiObjectiveOptimizer]],
    optimizer_kwargs_from_config: Dict[str, Any],
    max_evaluations: int,
    meta_data_seed: int,
) -> List[Evaluation]:
    """Run an optimization loop with the given `blackboxopt` optimizer on the provided
    benchmark instance until the maximum number of evaluations is reached.
    """
    main_objective = _prep_objective_and_optimizer_kwargs(benchmark=benchmark)

    if "meta_data" in inspect.signature(optimizer_cls).parameters.keys():
        optimizer_kwargs_from_config["meta_data"] = benchmark.get_meta_data(
            seed=meta_data_seed, distribution="random"
        )

    _, optimizer = _init_optimizer(
        benchmark=benchmark,
        optimizer_cls=optimizer_cls,
        objective=main_objective,
        optimizer_kwargs=optimizer_kwargs_from_config,
    )

    evaluations = sequential.run_optimization_loop(
        optimizer=optimizer,
        evaluation_function=benchmark,
        max_evaluations=max_evaluations,
    )

    return evaluations
