# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from copy import deepcopy
from typing import Dict, Optional

from blackboxopt import Evaluation

from scamlgp.benchmarking.benchmarks.api import SeedType
from scamlgp.benchmarking.noise.base import NoiseBase


class HomoscedasticGaussianNoise(NoiseBase):
    def __init__(self, noise_std: Dict[str, float], seed: Optional[SeedType] = None):
        """I.i.d. Gaussian noise with fixed scale as assumed in almost all GP like
        methods.

        Parameters
        ----------
        noise_std: Standard deviation(s) of the Gaussian Noise that is added directly to
            the noiseless benchmark objectives. The keys are the names objectives and
            the values correspond to the scale of the noise. Note that the dict must
            include all objectives for the Benchmark it is paired with, but can include
            additional keys that will not be used.
        seed: Random seed/number generator used for generating the noise.

        """
        super().__init__(seed)
        self.noise_std = noise_std

    def __call__(
        self,
        evaluation: Evaluation,
        rng=None,
    ) -> Evaluation:
        rng = self.rng if rng is None else rng

        tmp_eval = deepcopy(evaluation)
        for k in tmp_eval.objectives.keys():
            try:
                tmp_eval.objectives[k] += rng.normal(scale=self.noise_std[k])
            except KeyError:
                raise KeyError(
                    f"There is no noise for objective '{k}' defined! "
                    + "Please add a value to the noise_std parameter."
                )

        return tmp_eval

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(noise_std={self.noise_std}, seed={self._seed})"
        )
