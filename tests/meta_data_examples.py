from collections import defaultdict
from typing import Dict, Hashable, List

import numpy as np
import parameterspace as ps
from blackboxopt import Evaluation, EvaluationSpecification, Objective

META_DATA_MIXED_SPACE = {
    "task1": [
        Evaluation(
            configuration={
                "p1": 2,
                "p2": -0.1,
                "p3": 0.5,
                "p4": False,
                "p5": "small",
            },
            objectives={"loss": 1.0},
        ),
        Evaluation(
            configuration={
                "p1": 3,
                "p2": -0.2,
                "p3": 0.3,
                "p4": False,
                "p5": "medium",
            },
            objectives={"loss": 0.0},
        ),
    ],
    "task2": [
        Evaluation(
            configuration={
                "p1": 10,
                "p2": -0.3,
                "p3": 0.5,
                "p4": True,
                "p5": "medium",
            },
            objectives={"loss": 1.0},
        ),
        Evaluation(
            configuration={
                "p1": 12,
                "p2": -0.2,
                "p3": 0.3,
                "p4": False,
                "p5": "medium",
            },
            objectives={"loss": 1.0},
        ),
    ],
}

META_DATA_2D_SPACE = {
    "task1": [
        Evaluation(
            configuration={
                "x0": 2.5,
                "x1": -0.1,
            },
            objectives={"loss": 1.0},
        ),
        Evaluation(
            configuration={
                "x0": -2.5,
                "x1": -0.2,
            },
            objectives={"loss": 0.0},
        ),
    ],
    "task2": [
        Evaluation(
            configuration={
                "x0": 1,
                "x1": -0.3,
            },
            objectives={"loss": 1.0},
        ),
        Evaluation(
            configuration={
                "x0": 2,
                "x1": -0.2,
            },
            objectives={"loss": 1.0},
        ),
    ],
}

META_DATA_FIXED_PARAM_SPACE = {
    "task1": [
        Evaluation(
            configuration={"x": -1.0, "my_fixed_param": 1.0},
            objectives={"loss": 1.0 + 1},
        ),
        Evaluation(
            configuration={"x": 1.5, "my_fixed_param": 1.0},
            objectives={"loss": 2.25 + 1},
        ),
    ],
    "task2": [
        Evaluation(
            configuration={"x": 1.0, "my_fixed_param": 1.0},
            objectives={"loss": 1.0 / 2},
        ),
        Evaluation(
            configuration={"x": -1.5, "my_fixed_param": 1.0},
            objectives={"loss": 2.25 / 2},
        ),
    ],
}

META_DATA_CONDITIONAL_SPACE = {
    "task1": [
        Evaluation(
            configuration={"optimizer": "adam", "lr": 0.01},
            objectives={"loss": 0.6},
        ),
        Evaluation(
            configuration={"optimizer": "sgd", "lr": 0.01, "momentum": 0.5},
            objectives={"loss": 0.5},
        ),
    ]
}


META_DATA_1D_SPACE = {
    "task1": [
        Evaluation(
            configuration={"p1": 0.01},
            objectives={"loss": 0.6},
        ),
        Evaluation(
            configuration={"p1": 0.5},
            objectives={"loss": 0.5},
        ),
    ]
}


def forrester_fun(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Return the forrester function from the family specified by `a`, `b`, `c`"""
    original_forrester = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
    return a * original_forrester + b * x + c


def generate_forrester_meta_data(
    search_space: ps.ParameterSpace,
    objective: Objective,
    num_source_points: int,
    source_descriptors: List[dict],
) -> Dict[Hashable, List[Evaluation]]:
    """Helper to generate meta data for multiple tasks from the Forrester function.

    Args:
        search_space: Used for converting to/from the numerical representation of
            the configurations.
        objective: Used to invert the Forrester function values if
            `greater_is_better=True`
        num_source_points: The number of meta data points to generate per task.
        source_descriptors: For each source task, the descriptor kwargs.
    """
    meta_data = defaultdict(list)
    for i_task, source_descriptors in enumerate(source_descriptors):
        for _ in range(num_source_points):
            es = EvaluationSpecification(configuration=search_space.sample())
            loss = forrester_fun(
                x=search_space.to_numerical(es.configuration),
                **source_descriptors,
            )
            y = -1 * loss if objective.greater_is_better else loss
            e = es.create_evaluation({objective.name: float(y)})
            meta_data[i_task].append(e)

    return meta_data
