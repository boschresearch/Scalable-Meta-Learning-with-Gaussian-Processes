# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

"""Tests that can be imported and used to test optimizer implementations against this
package's blackbox optimizer interface.

This is an extension to `blackboxopt.optimizers.testing` with additional tests.
"""

import random
from typing import Type, Union

import numpy as np
import parameterspace as ps
from blackboxopt import Evaluation, Objective
from blackboxopt.base import MultiObjectiveOptimizer, SingleObjectiveOptimizer

META_DATA_1D = {
    "task_1": [
        Evaluation(configuration={"x0": 0.8}, objectives={"loss": -6.07}),
        Evaluation(configuration={"x0": 1.49}, objectives={"loss": -18.6}),
        Evaluation(configuration={"x0": 1.56}, objectives={"loss": -19.9}),
        Evaluation(configuration={"x0": 2.5}, objectives={"loss": -33.2}),
        Evaluation(configuration={"x0": 3.0}, objectives={"loss": -29.2}),
        Evaluation(configuration={"x0": 1.2}, objectives={"loss": -31.1}),
        Evaluation(configuration={"x0": 2.7}, objectives={"loss": -30.2}),
    ]
}


def _run_experiment_1d_deterministic(x0):
    _x = np.copy(np.atleast_2d(x0))
    params = np.array([0.75, 0.0, -10.0, 0.0, 0.0])
    y = np.polyval(params, _x)
    return float(np.squeeze(y))


def _run_optimizer(optimizer, steps=5):
    evaluations = []
    for _ in range(steps):
        es = optimizer.generate_evaluation_specification()
        evaluation = es.create_evaluation(
            objectives={"loss": _run_experiment_1d_deterministic(**es.configuration)}
        )
        optimizer.report(evaluation)
        evaluations.append(evaluation)
    return evaluations


def is_deterministic_with_shuffled_meta_data(
    optimizer_class: Union[
        Type[SingleObjectiveOptimizer], Type[MultiObjectiveOptimizer]
    ],
    optimizer_kwargs: dict,
    seed: int,
):
    """Runs the same optimization twice using the same evaluations as meta data, but in
    different orders. In both cases, the optimizer should still propose the exact same
    configurations during the optimization process.

    To ensure that the optimizer actually takes the meta data into account, we run the
    the same optimization a third time, but now use _different_ meta data and expect
    different configuration proposals.
    """

    if issubclass(optimizer_class, MultiObjectiveOptimizer):
        optimizer_kwargs["objectives"] = [Objective("loss", False)]

    if issubclass(optimizer_class, SingleObjectiveOptimizer):
        optimizer_kwargs["objective"] = Objective("loss", False)

    space = ps.ParameterSpace()
    space.add(ps.ContinuousParameter("x0", (0.5, 3)))
    space.seed(seed)

    # Run two times with shuffled meta data
    test_runs = []
    for _ in range(2):
        shuffled_data = META_DATA_1D.copy()
        for evals in shuffled_data.values():
            random.shuffle(evals)
        optimizer_kwargs["meta_data"] = shuffled_data

        optimizer = optimizer_class(space, seed=seed, **optimizer_kwargs)
        evaluations = _run_optimizer(optimizer)
        test_runs.append(evaluations)

    # Run one time with totally different meta data as reference
    optimizer_kwargs["meta_data"] = {
        "task_1": [Evaluation(configuration={"x0": 0.55}, objectives={"loss": -4.07})]
    }
    optimizer = optimizer_class(space, seed=seed, **optimizer_kwargs)
    evals_other_metadata = _run_optimizer(optimizer)

    x0s_other_metadata = [e.configuration["x0"] for e in evals_other_metadata]
    x0s_1 = [e.configuration["x0"] for e in test_runs[0]]
    x0s_2 = [e.configuration["x0"] for e in test_runs[1]]

    assert set(x0s_1) == set(x0s_2)
    assert set(x0s_other_metadata) != set(x0s_2)


META_OPTIMIZER_REFERENCE_TESTS = [is_deterministic_with_shuffled_meta_data]
