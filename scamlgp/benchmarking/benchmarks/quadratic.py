# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from typing import Optional

import numpy as np
import parameterspace as ps

from scamlgp.benchmarking.benchmarks.api import SeedType
from scamlgp.benchmarking.benchmarks.base import Base
from scamlgp.benchmarking.functions.quadratic import Quadratic as QuadraticFunction


class Quadratic(Base):
    def __init__(
        self, n_data_per_task=[4] * 128, seed: Optional[SeedType] = None, **kwargs
    ):
        """One-dimensional Quadratic function with parameters a, b, c.
        f(x) = a^2 * (x + b)^2 + c
        """
        prng = np.random.default_rng(seed)
        descriptors = ps.ParameterSpace()
        descriptors.add(ps.ContinuousParameter(name="a", bounds=[0.5, 1.5]))
        descriptors.add(ps.ContinuousParameter(name="b", bounds=[-0.9, 0.9]))
        descriptors.add(ps.ContinuousParameter(name="c", bounds=[-1, 1]))

        settings = ps.ParameterSpace()
        context = ps.ParameterSpace()

        search_space = ps.ParameterSpace()
        search_space.add(ps.ContinuousParameter(name="x", bounds=[-1, 1]))

        target_task, meta_tasks = super().create_tasks(
            descriptors, settings, context, len(n_data_per_task), prng
        )
        super().__init__(
            descriptors,
            settings,
            context,
            search_space,
            target_task,
            meta_tasks,
            n_data_per_task,
            **kwargs
        )

    @property
    def function(self):
        return QuadraticFunction()

    @property
    def optimum(self):
        return self.target_task.descriptors["c"]
