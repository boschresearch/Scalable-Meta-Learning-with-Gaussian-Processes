# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import parameterspace as ps
from blackboxopt import Evaluation, EvaluationSpecification, Objective

import scamlgp.benchmarking.benchmarks.api
from scamlgp.benchmarking.benchmarks.base import SeedType

HERE = Path(__file__).parent

_PARAMETER_NAMES = {
    "hps.lr_hparams.decay_steps_factor": "decay_steps_factor",
    "hps.lr_hparams.initial_value": "initial_value",
    "hps.lr_hparams.power": "power",
    "hps.opt_hparams.momentum": "momentum",
}


class PD1(scamlgp.benchmarking.benchmarks.api.Benchmark):
    def __init__(
        self,
        n_data_per_task: List[int] = None,
        target_task_id: Optional[str] = None,
        seed: Optional[SeedType] = None,
        **kwargs,
    ):
        self._n_data_per_task = [] if not n_data_per_task else n_data_per_task

        self._df: pd.DataFrame = pd.read_pickle(HERE / "pd1.pickle")
        self._df = self._df.rename(columns=_PARAMETER_NAMES)

        self._search_space = ps.ParameterSpace()
        self._search_space.add(
            ps.ContinuousParameter("decay_steps_factor", (0.01, 0.99))
        )
        self._search_space.add(
            ps.ContinuousParameter("initial_value", (np.log(1e-5), np.log(10)))
        )
        self._search_space.add(ps.ContinuousParameter("power", (0.1, 2.0)))
        self._search_space.add(
            ps.ContinuousParameter("momentum", (np.log(1e-3), np.log(1)))
        )

        self._objective = Objective("best_valid/error_rate", greater_is_better=False)

        self._prng = np.random.default_rng(seed)

        task_ids = list(self._df["study_group"].unique())
        task_ids.remove("imagenet_resnet50,imagenet,resnet,resnet50,1024")
        if target_task_id is not None:
            if target_task_id not in task_ids:
                raise ValueError(
                    f"Target task ID '{target_task_id}' needs to be one of {task_ids}"
                )
        else:
            target_task_id = self._prng.choice(task_ids)

        task_ids.remove(target_task_id)
        meta_task_ids = self._prng.choice(
            task_ids, size=len(self._n_data_per_task), replace=False
        )
        self._target_task = scamlgp.benchmarking.benchmarks.api.Task(
            uid=target_task_id,
            descriptors={"task_id": target_task_id},
            settings={},
            context={},
        )
        self._meta_tasks = {
            task_id: scamlgp.benchmarking.benchmarks.api.Task(
                uid=task_id, descriptors={"task_id": task_id}, settings={}, context={}
            )
            for task_id in meta_task_ids
        }

    def __call__(
        self,
        eval_spec: EvaluationSpecification,
        task_uid: Optional[Union[str, int]] = None,
    ) -> Evaluation:
        """Evaluate the benchmark function at the specified points.

        Parameters:
        -----------
        eval_spec
            The settings for all parameters for the evaluation. If present, the
            entries `settings` and `context` will overwrite the task's
            default values.
        task_uid: Unique identifier of the task. Defaults to the target task.

        Returns:
        --------
        Evaluation including the objective values and cost information in `user_info`
        """
        if task_uid is None:
            task_uid = self.target_task.uid

        task_df = self._df[self._df["study_group"] == task_uid]

        parameter_values = task_df[eval_spec.configuration.keys()].values
        absolute_differences = np.abs(
            parameter_values - np.array(list(eval_spec.configuration.values()))
        )
        i_closest = np.argmin(absolute_differences.sum(1))

        objective_value = task_df.iloc[i_closest][self.objective.name]

        return eval_spec.create_evaluation(
            objectives={self.objective.name: objective_value}
        )

    @property
    def objective(self) -> Objective:
        return self._objective

    @property
    def objectives(self) -> List[Objective]:
        return [self.objective]

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
        task_df = self._df[self._df["study_group"] == self.target_task.uid]
        minimum = task_df[self.objective.name].min()
        return minimum

    def get_meta_data(self, distribution="random", seed=None):
        if distribution != "random":
            raise NotImplementedError(
                f"Only random distribution is available, but got {distribution}."
            )

        meta_data = {}
        for task_id, n_task_data in zip(self.meta_tasks.keys(), self._n_data_per_task):
            task_df = self._df[self._df["study_group"] == task_id]
            meta_data[task_id] = [
                Evaluation(
                    configuration=row[
                        self.search_space.get_parameter_names()
                    ].to_dict(),
                    objectives={self.objective.name: float(row[self.objective.name])},
                )
                for _, row in task_df.sample(n=n_task_data, replace=False).iterrows()
            ]
        return meta_data
