# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from typing import Optional

import parameterspace as ps

from scamlgp.benchmarking.benchmarks.api import SeedType
from scamlgp.benchmarking.benchmarks.base import get_minimum
from scamlgp.benchmarking.benchmarks.hartmann_3d import Hartmann3D
from scamlgp.benchmarking.functions.hartmann import Hartmann6D as Hartmann6DFunction


class Hartmann6D(Hartmann3D):
    """The six-dimensional Hartmann function.

    The function has six local minima and one global minimum.
    For more information see the function documentation.

    Reference: https://www.sfu.ca/~ssurjano/hart6.html

    The range of values for `alpha1` ... `alpha4` were taken in accordance with
    https://github.com/amzn/emukit/commit/a46d254064e575f2b61a6378e2819494849c147a#diff-43b88d268e27ef759342c3b35be8228e
    """

    def __init__(
        self, n_data_per_task=[4] * 128, seed: Optional[SeedType] = None, **kwargs
    ):
        super().__init__(n_data_per_task, seed=seed, **kwargs)

        self._search_space.add(ps.ContinuousParameter(name="x4", bounds=[0, 1]))
        self._search_space.add(ps.ContinuousParameter(name="x5", bounds=[0, 1]))
        self._search_space.add(ps.ContinuousParameter(name="x6", bounds=[0, 1]))

    @property
    def function(self):
        return Hartmann6DFunction()

    @property
    def optimum(self):
        return get_minimum(self)
