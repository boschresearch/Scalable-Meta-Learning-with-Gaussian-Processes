# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import json
from typing import Dict, List, Optional, Union

import numpy as np
import parameterspace as ps
from blackboxopt import Evaluation, EvaluationSpecification, Objective
from ConfigSpace.read_and_write import json as config_space_json
from hpobench.benchmarks.ml.tabular_benchmark import (
    TabularBenchmark as OriginalHPOBenchTabularBenchmark,
)
from parameterspace.configspace_utils import parameterspace_from_configspace_dict
from scipy.stats.qmc import Sobol

import scamlgp.benchmarking.benchmarks.api
from scamlgp.benchmarking.benchmarks.api import SeedType

TASK_IDS = {
    "xgb": [
        "10101",
        "12",
        "146212",
        "146606",
        "146818",
        "146821",
        "146822",
        "14965",
        "167119",
        "167120",
        "168911",
        "168912",
        "3",
        "31",
        "3917",
        "53",
        "7592",
        "9952",
        "9977",
        "9981",
    ],
    "rf": [
        "10101",
        "12",
        "146195",
        "146212",
        "146606",
        "146818",
        "146821",
        "146822",
        "14965",
        "167119",
        "167120",
        "168329",
        "168330",
        "168331",
        "168335",
        "168868",
        "168908",
        "168910",
        "168911",
        "168912",
        "3",
        "31",
        "3917",
        "53",
        "7592",
        "9952",
        "9977",
        "9981",
    ],
    "lr": [
        "10101",
        "146195",
        "146606",
        "146821",
        "14965",
        "167120",
        "168330",
        "168335",
        "168908",
        "168910",
        "168912",
        "31",
        "53",
        "9952",
        "9981",
        "12",
        "146212",
        "146818",
        "146822",
        "167119",
        "168329",
        "168331",
        "168868",
        "168909",
        "168911",
        "3",
        "3917",
        "7592",
        "9977",
    ],
    "svm": [
        "10101",
        "146195",
        "146606",
        "146821",
        "14965",
        "167120",
        "168330",
        "168335",
        "168908",
        "168910",
        "168912",
        "31",
        "53",
        "9952",
        "9981",
        "12",
        "146212",
        "146818",
        "146822",
        "167119",
        "168329",
        "168331",
        "168868",
        "168909",
        "168911",
        "3",
        "3917",
        "7592",
        "9977",
    ],
    "nn": ["10101", "146818", "146821", "146822", "31", "3917", "53", "9952"],
}


class HPOBenchTabular(scamlgp.benchmarking.benchmarks.api.Benchmark):
    def __init__(
        self,
        scenario: str,
        n_data_per_task: Optional[List[int]] = None,
        target_task_id: Optional[str] = None,
        data_dir: Optional[str] = None,
        seed: Optional[SeedType] = None,
    ):
        """A wrapper for HPOBench's ML TabularBenchmark interface, ready to generate
        meta-data based on different OpenML tasks, but not yet exposing the available
        settings space for multi-fidelity optimization.

        NOTE: Make sure that your acquisition function leverages or is at least robust
        to this being a tabular benchmark only exposing ordinal parameters; just using
        it via the default parameterspace numerical representations can cause issues.

        Args:
            scenario: One of "lr", "nn", "rf", "svm", "xgb"
            n_data_per_task: The length of this list corresponds to the number of tasks,
                and each entry to the number of data points per that task.
            target_task_id: One of `TASK_IDS[scenario]`. Sampled randomly by default.
            data_dir: Path to the downloaded tabular data. Downloaded automatically if
                not given.
            seed: Random seed for reproducibility
        """
        if scenario not in TASK_IDS:
            raise ValueError(
                f"Scenario '{scenario}' needs to be one of {list(TASK_IDS.keys())}"
            )
        self._scenario = scenario
        self._data_dir = data_dir
        self.objectives = [Objective("1 - Accuracy", greater_is_better=False)]
        self._n_data_per_task = [] if n_data_per_task is None else n_data_per_task

        self._prng = np.random.default_rng(seed)

        task_ids = TASK_IDS[self._scenario].copy()
        if target_task_id is not None:
            if target_task_id not in task_ids:
                raise ValueError(
                    f"Target task ID '{target_task_id}' needs to be one of {task_ids}"
                )
        else:
            target_task_id = self._prng.choice(task_ids)

        task_ids.remove(target_task_id)
        meta_task_ids = self._prng.choice(
            task_ids, size=len(n_data_per_task), replace=False
        )
        self._target_task = scamlgp.benchmarking.benchmarks.api.Task(
            uid=target_task_id,
            descriptors={"task_id": target_task_id},
            settings={},
            context={},
        )
        self._meta_tasks = {
            i: scamlgp.benchmarking.benchmarks.api.Task(
                uid=i, descriptors={"task_id": i}, settings={}, context={}
            )
            for i in meta_task_ids
        }

        self._target_task_benchmark = OriginalHPOBenchTabularBenchmark(
            model=self._scenario,
            task_id=self.target_task.uid,
            data_dir=self._data_dir,
            rng=seed,
        )

        self._search_space, names_to_map = parameterspace_from_configspace_dict(
            json.loads(
                config_space_json.write(
                    self._target_task_benchmark.get_configuration_space()
                )
            )
        )
        if names_to_map:
            raise NotImplementedError(
                "Search space contains parameter names that need to be mapped for "
                + f"ParameterSpace compatibility {names_to_map}"
            )

        # NOTE: No multi-fidelity support here
        self._default_fidelities = dict(
            self._target_task_benchmark.get_fidelity_space().get_default_configuration()
        )

    @property
    def target_task(self):
        return self._target_task

    @property
    def meta_tasks(self):
        return self._meta_tasks

    @property
    def search_space(self) -> ps.ParameterSpace:
        return self._search_space

    @property
    def output_dimensions(self) -> int:
        return len(self.objectives)

    @property
    def optimum(self) -> float:
        """Average `function_value` across available seeds at default fidelity from
        underlying look up table and then taking the minimum of these averages.
        """
        df = self._target_task_benchmark.table

        # Select subset of data with matching fidelity
        for key, value in self._default_fidelities.items():
            # https://stackoverflow.com/a/46165056/8363967
            df = df[df[key].values == value]

        df = df.assign(
            function_value=[d["function_value"] for d in df["result"].values]
        )

        return (
            df.groupby(
                self.search_space.get_parameter_names()
                + self.search_space.get_constant_names(),
                as_index=False,
            )
            .mean(numeric_only=True)["function_value"]
            .min()
        )

    def __call__(
        self,
        eval_spec: EvaluationSpecification,
        task_uid: Optional[Union[str, int]] = None,
    ) -> Evaluation:
        if task_uid is not None and task_uid not in TASK_IDS[self._scenario]:
            raise ValueError(
                f"Task ID '{task_uid}' needs to be one of {TASK_IDS[self._scenario]}"
            )

        if task_uid is None:
            task_uid = self.target_task.uid

        benchmark = (
            self._target_task_benchmark
            if task_uid == self.target_task.uid
            else OriginalHPOBenchTabularBenchmark(
                model=self._scenario,
                task_id=task_uid,
                data_dir=self._data_dir,
                rng=self._prng.bit_generator.random_raw(),
            )
        )

        result = benchmark(
            configuration=eval_spec.configuration, fidelity=self._default_fidelities
        )
        return eval_spec.create_evaluation({self.objectives[0].name: result})

    def get_meta_data(
        self, seed: Optional[SeedType] = None, distribution: str = "random"
    ) -> Dict[Union[str, int], List[Evaluation]]:
        """Return data for pre-training. Output is data sampled from each meta
        task. Evaluations at `n_data_per_task` random points within the search space.

        Args:
            seed: Seed for random generator to get the same metadata
            distribution: Controls the distribution of the datapoints for the task.
                Allowed values are "random" and "sobol".
        """
        if distribution not in ["random", "sobol"]:
            raise ValueError(
                f"Distribution for meta data generation {distribution} needs to be "
                + "one of 'random' or 'sobol'."
            )

        prng = np.random.default_rng(seed)
        sobol = Sobol(d=len(self.search_space), scramble=True, seed=seed)

        meta_data: Dict[Union[str, int], List[Evaluation]] = dict()
        for (uid, _), n_data in zip(self.meta_tasks.items(), self._n_data_per_task):
            meta_data[uid] = []
            benchmark = OriginalHPOBenchTabularBenchmark(
                model=self._scenario, task_id=uid, data_dir=self._data_dir, rng=seed
            )

            for _ in range(n_data):
                if distribution == "random":
                    config = self.search_space.sample(rng=prng)
                else:
                    vector = sobol.random().flatten()
                    config = self.search_space.from_numerical(vector)

                result = benchmark(configuration=config)

                meta_data[uid].append(
                    Evaluation(
                        configuration=config,
                        objectives={self.objectives[0].name: result},
                    )
                )

            del benchmark

        return meta_data
