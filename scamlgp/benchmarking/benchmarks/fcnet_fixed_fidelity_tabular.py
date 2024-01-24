# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import json
import tarfile
import tempfile
import urllib.request
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np
import parameterspace as ps
from blackboxopt import Evaluation, EvaluationSpecification, Objective

from scamlgp.benchmarking.benchmarks.api import SeedType

_DATASET_URL = (
    "https://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz"
)

_SEARCH_SPACE_JSON = '{"parameters": {"activation_fn_1": {"parameter": {"class_name": "parameterspace.parameters.categorical.CategoricalParameter", "init_args": [], "init_kwargs": {"name": "activation_fn_1", "values": ["tanh", "relu"]}, "transformation": {"class_name": "parameterspace.transformations.categorical.Cat2Num", "init_args": [["tanh", "relu"]], "init_kwargs": {}}, "prior": {"class_name": "parameterspace.priors.categorical.Categorical", "init_args": [[1.0, 1.0]], "init_kwargs": {}}}, "condition": {}}, "activation_fn_2": {"parameter": {"class_name": "parameterspace.parameters.categorical.CategoricalParameter", "init_args": [], "init_kwargs": {"name": "activation_fn_2", "values": ["tanh", "relu"]}, "transformation": {"class_name": "parameterspace.transformations.categorical.Cat2Num", "init_args": [["tanh", "relu"]], "init_kwargs": {}}, "prior": {"class_name": "parameterspace.priors.categorical.Categorical", "init_args": [[1.0, 1.0]], "init_kwargs": {}}}, "condition": {}}, "batch_size": {"parameter": {"class_name": "parameterspace.parameters.ordinal.OrdinalParameter", "init_args": [], "init_kwargs": {"name": "batch_size", "values": [8, 16, 32, 64]}, "transformation": {"class_name": "parameterspace.transformations.categorical.Cat2Num", "init_args": [[8, 16, 32, 64]], "init_kwargs": {}}, "prior": {"class_name": "parameterspace.priors.uniform.Uniform", "init_args": [], "init_kwargs": {}}}, "condition": {}}, "dropout_1": {"parameter": {"class_name": "parameterspace.parameters.ordinal.OrdinalParameter", "init_args": [], "init_kwargs": {"name": "dropout_1", "values": [0.0, 0.3, 0.6]}, "transformation": {"class_name": "parameterspace.transformations.categorical.Cat2Num", "init_args": [[0.0, 0.3, 0.6]], "init_kwargs": {}}, "prior": {"class_name": "parameterspace.priors.uniform.Uniform", "init_args": [], "init_kwargs": {}}}, "condition": {}}, "dropout_2": {"parameter": {"class_name": "parameterspace.parameters.ordinal.OrdinalParameter", "init_args": [], "init_kwargs": {"name": "dropout_2", "values": [0.0, 0.3, 0.6]}, "transformation": {"class_name": "parameterspace.transformations.categorical.Cat2Num", "init_args": [[0.0, 0.3, 0.6]], "init_kwargs": {}}, "prior": {"class_name": "parameterspace.priors.uniform.Uniform", "init_args": [], "init_kwargs": {}}}, "condition": {}}, "init_lr": {"parameter": {"class_name": "parameterspace.parameters.ordinal.OrdinalParameter", "init_args": [], "init_kwargs": {"name": "init_lr", "values": [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]}, "transformation": {"class_name": "parameterspace.transformations.categorical.Cat2Num", "init_args": [[0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]], "init_kwargs": {}}, "prior": {"class_name": "parameterspace.priors.uniform.Uniform", "init_args": [], "init_kwargs": {}}}, "condition": {}}, "lr_schedule": {"parameter": {"class_name": "parameterspace.parameters.categorical.CategoricalParameter", "init_args": [], "init_kwargs": {"name": "lr_schedule", "values": ["cosine", "const"]}, "transformation": {"class_name": "parameterspace.transformations.categorical.Cat2Num", "init_args": [["cosine", "const"]], "init_kwargs": {}}, "prior": {"class_name": "parameterspace.priors.categorical.Categorical", "init_args": [[1.0, 1.0]], "init_kwargs": {}}}, "condition": {}}, "n_units_1": {"parameter": {"class_name": "parameterspace.parameters.ordinal.OrdinalParameter", "init_args": [], "init_kwargs": {"name": "n_units_1", "values": [16, 32, 64, 128, 256, 512]}, "transformation": {"class_name": "parameterspace.transformations.categorical.Cat2Num", "init_args": [[16, 32, 64, 128, 256, 512]], "init_kwargs": {}}, "prior": {"class_name": "parameterspace.priors.uniform.Uniform", "init_args": [], "init_kwargs": {}}}, "condition": {}}, "n_units_2": {"parameter": {"class_name": "parameterspace.parameters.ordinal.OrdinalParameter", "init_args": [], "init_kwargs": {"name": "n_units_2", "values": [16, 32, 64, 128, 256, 512]}, "transformation": {"class_name": "parameterspace.transformations.categorical.Cat2Num", "init_args": [[16, 32, 64, 128, 256, 512]], "init_kwargs": {}}, "prior": {"class_name": "parameterspace.priors.uniform.Uniform", "init_args": [], "init_kwargs": {}}}, "condition": {}}}, "constants": {}}'  # noqa

_OPTIMAL_CONFIGS_WITH_MEAN_VALID_LOSS = {
    "fcnet_slice_localization_data": Evaluation(
        configuration={
            "activation_fn_1": "relu",
            "activation_fn_2": "tanh",
            "batch_size": 16,
            "dropout_1": 0.0,
            "dropout_2": 0.0,
            "init_lr": 0.0005,
            "lr_schedule": "cosine",
            "n_units_1": 256,
            "n_units_2": 512,
        },
        objectives={"valid_loss": 0.00019159916337230243},
    ),
    "fcnet_protein_structure_data": Evaluation(
        configuration={
            "activation_fn_1": "relu",
            "activation_fn_2": "relu",
            "batch_size": 8,
            "dropout_1": 0.0,
            "dropout_2": 0.3,
            "init_lr": 0.0005,
            "lr_schedule": "cosine",
            "n_units_1": 512,
            "n_units_2": 512,
        },
        objectives={"valid_loss": 0.221378855407238},
    ),
    "fcnet_naval_propulsion_data": Evaluation(
        configuration={
            "activation_fn_1": "tanh",
            "activation_fn_2": "relu",
            "batch_size": 8,
            "dropout_1": 0.0,
            "dropout_2": 0.0,
            "init_lr": 0.0005,
            "lr_schedule": "cosine",
            "n_units_1": 128,
            "n_units_2": 512,
        },
        objectives={"valid_loss": 3.19113473778998e-05},
    ),
    "fcnet_parkinsons_telemonitoring_data": Evaluation(
        configuration={
            "activation_fn_1": "relu",
            "activation_fn_2": "relu",
            "batch_size": 8,
            "dropout_1": 0.0,
            "dropout_2": 0.0,
            "init_lr": 0.005,
            "lr_schedule": "cosine",
            "n_units_1": 32,
            "n_units_2": 512,
        },
        objectives={"valid_loss": 0.0067059280117973685},
    ),
}


def _download_and_extract_look_up_tables(
    target_directory: Optional[PathLike] = None,
) -> Path:
    tmp_download_target = Path(tempfile.gettempdir(), "fcnet_tabular_benchmarks.tar.gz")

    if not tmp_download_target.exists():
        urllib.request.urlretrieve(_DATASET_URL, str(tmp_download_target))

    if target_directory is None:
        target_directory = Path(tempfile.mkdtemp(prefix="bbo_bench_fcnet_"))

    with tarfile.open(tmp_download_target) as fh:
        fh.extractall(target_directory)

    return target_directory / "fcnet_tabular_benchmarks"


def _load_look_up_table(
    hdf5_lut_file: PathLike,
    metric_name: str = "valid_loss",
    i_epoch: int = -1,
    i_seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Args:
        hdf5_lut_file: File path to the look up table's hdf5 file.
        metric_name: Name of the target metric to return.
        i_epoch: The datasets usually contain evaluations for 100 epochs, pick one.
        i_seed: When given the metric for that (usually one of four) seeded runs is
            returned. Otherwise, the mean metric value across all seeds is returned.
    """
    lut = {}
    with h5py.File(hdf5_lut_file) as fh:
        for k, v in fh.items():
            if i_seed is None:
                lut[k] = float(np.mean(v[metric_name][()][:, i_epoch]))
            else:
                lut[k] = float(v[metric_name][()][i_seed, i_epoch])
    return lut


class FCNetFixedFidelityTabularBenchmark:
    def __init__(
        self,
        target_task_file: str,
        meta_task_files: Optional[List[str]] = None,
        n_data_per_task: Optional[List[int]] = None,
        fix_search_space: Optional[Dict[str, Any]] = None,
        lazy_load_target_task_lut: bool = True,
        seed: Optional[SeedType] = None,
    ) -> None:
        """A light wrapper of the tabular benchmark data from
        https://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz

        Args:
            target_task_file: File path to target task look up table.
            meta_task_files: File path to meta task look up tables.
            n_data_per_task: How many data points per task should be contained in the
                meta data.
            fix_search_space: Parameter name to fixed value mapping that should be
                applied to the search space.
            lazy_load_target_task_lut: Whether to load the target task look up table on
                the first evaluation of the benchmark instead of during initialization.
                This can help keep the peak memory footprint down.
            seed: Seed for meta data generation and other potential randomness.
        """
        if meta_task_files is None:
            meta_task_files = []
        if n_data_per_task is None:
            n_data_per_task = []

        assert len(meta_task_files) == len(
            n_data_per_task
        ), "Meta tasks need to match data per task specification"

        self._objective = Objective("valid_loss", greater_is_better=False)
        self._meta_task_files = meta_task_files
        self._n_data_per_task = n_data_per_task

        self._fix_search_space = fix_search_space
        self._search_space = ps.ParameterSpace.from_dict(json.loads(_SEARCH_SPACE_JSON))
        self._search_space.seed(seed)
        self._search_space.fix(**self._fix_search_space)

        self._target_task_file = target_task_file
        if not Path(self._target_task_file).exists():
            print(
                "Downloading tabular dataset, because no file found at",
                self._target_task_file,
            )
            download_directory = _download_and_extract_look_up_tables()
            self._target_task_file = download_directory / self._target_task_file
            self._meta_task_files = [
                str(download_directory / mtf) for mtf in self._meta_task_files
            ]

        self._target_task_lut = (
            None
            if lazy_load_target_task_lut
            else _load_look_up_table(
                hdf5_lut_file=self._target_task_file, metric_name=self._objective.name
            )
        )

    @property
    def objective(self) -> Objective:
        return self._objective

    @property
    def objectives(self) -> List[Objective]:
        return [self.objective]

    @property
    def optimum(self) -> float:
        return _OPTIMAL_CONFIGS_WITH_MEAN_VALID_LOSS[
            Path(self._target_task_file).stem
        ].objectives[self._objective.name]

    @property
    def output_dimensions(self) -> int:
        return 1

    @property
    def search_space(self) -> ps.ParameterSpace:
        return self._search_space

    def __call__(
        self,
        eval_spec: EvaluationSpecification,
        task_uid: Optional[Union[str, int]] = None,
    ) -> Evaluation:
        if task_uid is not None:
            raise NotImplementedError("No support for custom target task IDs.")

        # Lazy load target task look up table to reduce peak memory footprint
        if self._target_task_lut is None:
            self._target_task_lut = _load_look_up_table(
                hdf5_lut_file=self._target_task_file, metric_name=self._objective.name
            )

        # Sort the dictionary keys alphabetically to ensure that the look up works
        config = {
            k: eval_spec.configuration[k]
            for k in sorted(eval_spec.configuration.keys())
        }
        objective_value = self._target_task_lut[str(config).replace("'", '"')]

        return eval_spec.create_evaluation(
            objectives={self.objective.name: objective_value}
        )

    def get_meta_data(
        self, distribution: str = "random", seed: Optional[SeedType] = None
    ) -> Dict[Union[str, int], List[Evaluation]]:
        if distribution != "random":
            raise NotImplementedError(f"Distribution {distribution} is unavailable.")

        meta_data = {}
        for i_task, meta_task_file in enumerate(self._meta_task_files):
            bm = FCNetFixedFidelityTabularBenchmark(
                target_task_file=meta_task_file,
                fix_search_space=self._fix_search_space,
                seed=seed,
            )
            meta_data[Path(meta_task_file).stem] = [
                bm(EvaluationSpecification(bm.search_space.sample()))
                for _ in range(self._n_data_per_task[i_task])
            ]
        return meta_data


if __name__ == "__main__":
    file_paths = [
        "fcnet_naval_propulsion_data.hdf5",
        "fcnet_parkinsons_telemonitoring_data.hdf5",
        "fcnet_protein_structure_data.hdf5",
        "fcnet_slice_localization_data.hdf5",
    ]

    b = FCNetFixedFidelityTabularBenchmark(
        target_task_file=file_paths[-1],
        meta_task_files=file_paths[:-1],
        n_data_per_task=[512] * 3,
        fix_search_space={
            "activation_fn_1": "relu",
            "activation_fn_2": "relu",
            "lr_schedule": "cosine",
        },
    )
    md = b.get_meta_data()

    print("done")
