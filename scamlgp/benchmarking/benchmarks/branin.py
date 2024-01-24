# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from typing import List, Optional

import numpy as np
import parameterspace as ps

from scamlgp.benchmarking.benchmarks.api import SeedType
from scamlgp.benchmarking.benchmarks.base import Base, get_minimum
from scamlgp.benchmarking.functions.branin import Branin as BraninFunction


class Branin(Base):
    """The two-dimensional Branin function.

    The function is multi-modal and has three global minima. For more information see
    the function documentation.

    Reference: https://www.sfu.ca/~ssurjano/branin.html
    """

    def __init__(
        self,
        n_data_per_task: List[int] = [4] * 128,
        seed: Optional[SeedType] = None,
        **kwargs
    ):
        prng = np.random.default_rng(seed)

        # Randomly distributed [a, b, c, r, s, t] between descriptors, settings and
        # context
        descriptors = ps.ParameterSpace()
        descriptors.add(ps.ContinuousParameter(name="a", bounds=[0.5, 1.5]))
        descriptors.add(ps.ContinuousParameter(name="b", bounds=[0.1, 0.15]))
        descriptors.add(ps.ContinuousParameter(name="c", bounds=[1, 2]))

        settings = ps.ParameterSpace()
        settings.add(ps.ContinuousParameter(name="r", bounds=[5, 7]))
        settings.add(ps.ContinuousParameter(name="s", bounds=[8, 12]))

        context = ps.ParameterSpace()
        context.add(ps.ContinuousParameter(name="t", bounds=[0.03, 0.05]))

        search_space = ps.ParameterSpace()
        search_space.add(ps.ContinuousParameter(name="x1", bounds=[-5, 10]))
        search_space.add(ps.ContinuousParameter(name="x2", bounds=[0, 15]))

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
        return BraninFunction()

    @property
    def optimum(self):
        return get_minimum(self)
