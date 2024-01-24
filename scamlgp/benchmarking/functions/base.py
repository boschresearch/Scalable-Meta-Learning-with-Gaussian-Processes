# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import abc
from typing import Tuple, Union


class Base(abc.ABC):
    @abc.abstractmethod
    def __call__(self, **kwargs) -> Union[float, Tuple[float]]:
        """Evaluate the function at the specified points.

        Parameters:
        -----------
        kwargs: Set of parameter values of the function, input space included.

        Returns:
        --------
            Observed value(s) at the query point.
        """
