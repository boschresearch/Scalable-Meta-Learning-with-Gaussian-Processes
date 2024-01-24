# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import math

from scamlgp.benchmarking.functions.base import Base


class Branin(Base):
    def __call__(  # type: ignore[override]
        self,
        x1: float,
        x2: float,
        a: float = 1,
        b: float = 5.1 / (4 * math.pi**2),
        c: float = 5 / math.pi,
        r: float = 6,
        s: float = 10,
        t: float = 1 / (8 * math.pi),
    ) -> float:
        r"""The two-dimensional Branin function.

        The function is a multimodal function and has three global minima.
        The function is given by

        .. math::
        f(x) = a\cdot(x_2-b \cdot x_1^2+c \cdot x_1-d)^2+e\cdot(1-f)\cdot\cos(x_1)+e

        Reference: https://www.sfu.ca/~ssurjano/branin.html

        Parameters:
        -----------
        x1, x2: Values for the two input parameters.
        a, c, d, f: floats
            The parameters for the Branin function.

        Returns:
        --------
            Observed value at the query points.
        """
        y = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s
        return y
