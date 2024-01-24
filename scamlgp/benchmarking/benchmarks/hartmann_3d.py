# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from typing import Optional

import numpy as np
import parameterspace as ps

from scamlgp.benchmarking.benchmarks.api import SeedType
from scamlgp.benchmarking.benchmarks.base import Base, get_minimum
from scamlgp.benchmarking.functions.hartmann import Hartmann3D as Hartmann3DFunction


class Hartmann3D(Base):
    """The three-dimensional Hartmann function.

    The function has four local minima and one global minimum. For more information see
    the function documentation.

    Reference: https://www.sfu.ca/~ssurjano/hart3.html

    The range of values for `alpha1` ... `alpha4` were taken in accordance with
    https://github.com/amzn/emukit/commit/a46d254064e575f2b61a6378e2819494849c147a#diff-43b88d268e27ef759342c3b35be8228e
    """

    def __init__(
        self, n_data_per_task=[4] * 128, seed: Optional[SeedType] = None, **kwargs
    ):
        prng = np.random.default_rng(seed)
        descriptors = ps.ParameterSpace()
        descriptors.add(ps.ContinuousParameter(name="alpha1", bounds=[1.0, 1.02]))
        descriptors.add(ps.ContinuousParameter(name="alpha2", bounds=[1.18, 1.2]))
        descriptors.add(ps.ContinuousParameter(name="alpha3", bounds=[2.8, 3.0]))
        descriptors.add(ps.ContinuousParameter(name="alpha4", bounds=[3.2, 3.4]))

        settings = ps.ParameterSpace()
        context = ps.ParameterSpace()

        search_space = ps.ParameterSpace()
        search_space.add(ps.ContinuousParameter(name="x1", bounds=[0, 1]))
        search_space.add(ps.ContinuousParameter(name="x2", bounds=[0, 1]))
        search_space.add(ps.ContinuousParameter(name="x3", bounds=[0, 1]))

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
        return Hartmann3DFunction()

    @property
    def optimum(self):
        return get_minimum(self)
