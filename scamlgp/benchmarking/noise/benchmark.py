# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from typing import Dict, List, Optional, Union

import numpy as np
from blackboxopt import Evaluation, EvaluationSpecification, Objective
from parameterspace import ParameterSpace

from scamlgp.benchmarking.benchmarks.api import SeedType, Task
from scamlgp.benchmarking.benchmarks.base import Base
from scamlgp.benchmarking.noise.base import NoiseBase


class NoisyBenchmark(Base):
    def __init__(self, benchmark: Base, noise_model: NoiseBase):
        """Combine a Benchmark and a Noise into a single NoisyBenchmark."""
        self.noise_free_benchmark = benchmark
        self.noise_model = noise_model
        if hasattr(self.noise_free_benchmark, "optimum"):
            self.optimum = self.noise_free_benchmark.optimum  # type: ignore
        if hasattr(self.noise_free_benchmark, "pareto_front"):
            self.pareto_front = self.noise_free_benchmark.pareto_front  # type: ignore

    @property
    def target_task(self) -> Task:
        return self.noise_free_benchmark._target_task

    @property
    def meta_tasks(self) -> Dict[Union[str, int], Task]:
        return self.noise_free_benchmark._meta_tasks

    @property
    def search_space(self) -> ParameterSpace:
        return self.noise_free_benchmark._search_space

    @property
    def output_dimensions(self) -> int:
        return len(self.noise_free_benchmark.objectives)

    @property
    def objectives(self) -> List[Objective]:
        return self.noise_free_benchmark._objectives

    def __call__(
        self,
        eval_spec: EvaluationSpecification,
        task_uid: Optional[Union[str, int]] = None,
    ) -> Evaluation:
        """Calls noise-free benchmark and applies the noise model to the Evaluation."""
        evaluation = self.noise_free_benchmark(eval_spec=eval_spec, task_uid=task_uid)
        evaluation = self.noise_model(evaluation)
        return evaluation

    def get_meta_data(
        self, distribution: str, seed: Optional[SeedType] = None
    ) -> Dict[Union[str, int], List[Evaluation]]:
        """Generate data from noiseless benchmark and apply noise to each evaluation.

        Args:
            distribution: "random", "sobol" or what ever the underlying noise free
            benchmark supports
            seed: Random seed for reproducibility
        """
        rng = np.random.default_rng(seed)

        noise_free_meta_data = self.noise_free_benchmark.get_meta_data(
            seed=rng, distribution=distribution
        )

        noisy_meta_data = {
            task_id: [self.noise_model(e, rng) for e in eval_list]
            for task_id, eval_list in noise_free_meta_data.items()
        }

        return noisy_meta_data
