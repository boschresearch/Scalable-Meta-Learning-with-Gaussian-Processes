import abc
from typing import Optional

import numpy as np
from blackboxopt import Evaluation

from scamlgp.benchmarking.benchmarks.api import SeedType


class NoiseBase:
    def __init__(self, seed: Optional[SeedType] = None):
        """Minimal base class for different noise types.

        For convenience, the base class handles the seed and stores a random number
        generator as an attribute.

        Parameter
        ---------
        seed: Seed to initialize the random number generator that can be used in any
            derivative class.
        """
        self._seed = seed
        self.rng = np.random.default_rng(self._seed)

    @abc.abstractmethod
    def __call__(
        self,
        evaluation: Evaluation,
        rng: Optional[np.random.Generator] = None,
    ) -> Evaluation:
        """Apply noise to an Evaluation.

        Parameters
        ----------
        evaluation: A valid evaluation with a configuration and objectives.
        rng: A random number generator to use. If None, the internal one is used.

        Returns
        -------

        A new Evaluation object with the noise applied to it.
        """
