# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import math

from scamlgp.benchmarking.functions.base import Base


class Quadratic(Base):
    """One-dimensional Quadratic function with parameters a, b, c.
    f(x) = a^2 * (x + b)^2 + c
    """

    def __call__(self, x: float, a: float, b: float, c: float) -> float:  # type: ignore[override]
        """Evaluate the quadratic function at the specified points.

        Parameters:
        -----------
        x: Numerical representation of the points.
        a, b, c: The parameters for the quadratic function.
                a^2 * (x+b)^2 + c

        Returns:
        --------
        tuple (numpy.array, numpy.array)
            Observed value at the query points and the associated costs towards the
            total budget of an optimizer.
        """
        return math.pow(a * (x + b), 2) + c
