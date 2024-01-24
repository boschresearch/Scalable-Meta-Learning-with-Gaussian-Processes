# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import argparse
import copy
import glob
import inspect
import json
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from blackboxopt import Evaluation

import scamlgp.benchmarking.benchmarks
from scamlgp.benchmarking.benchmarks.api import Benchmark, SeedType
from scamlgp.benchmarking.experiment_config_utils import (
    Experiment,
    get_experiments_config_from_module,
    hash_experiment_config,
)
from scamlgp.benchmarking.local_runner import main as submit_local_job

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()


def get_benchmarks() -> List[Benchmark]:
    """
    Collect all benchmark classes that are directly importable from benchmarks_multitask
    and follow the benchmarks API.
    """
    importables = [
        getattr(scamlgp.benchmarking.benchmarks, name)
        for name in dir(scamlgp.benchmarking.benchmarks)
    ]

    def is_benchmark(ref):
        return inspect.isclass(ref) and issubclass(ref, Benchmark)

    benchmarks = list(filter(is_benchmark, importables))
    return benchmarks


def get_benchmarks_with_search_space_dimensions(dimensions: int) -> List[Benchmark]:
    """
    Collect a list of available benchmarks with a specific number of dimensions in
    its search space.

    Parameters
    ----------
    dimensions:
        Number of dimensions to check.

    Returns
    -------
    List of benchmarks with the specified number of search space dimensions.
    """
    benchmarks = get_benchmarks()

    def dimensions_filter(benchmark):
        try:
            return len(getattr(benchmark(), "search_space")) == dimensions
        except TypeError:
            logging.warning(f"Could not instantiate benchmark '{benchmark}'.")
            return False

    return list(filter(dimensions_filter, benchmarks))


def get_benchmarks_with_output_dimensions(dimensions: int) -> List[Benchmark]:
    """
    Collect a list of available benchmarks with a specific number of dimensions in
    its output space.

    Parameters
    ----------
    dimensions:
        Number of dimensions to check.

    Returns
    -------
    List of benchmarks with the specified number of output dimensions.
    """
    benchmarks = get_benchmarks()

    def dimensions_filter(benchmark):
        try:
            return benchmark().output_dimensions == dimensions
        except TypeError:
            logging.warning(f"Could not instantiate benchmark '{benchmark}'.")
            return False

    return list(filter(dimensions_filter, benchmarks))


def add_noise_to_meta_data_objectives(
    meta_data: Dict[Union[str, int], List[Evaluation]],
    noise_scale: Union[float, Dict[str, float]],
    noise_is_relative: bool = False,
    seed: Optional[SeedType] = None,
) -> Dict[Union[str, int], List[Evaluation]]:
    """Add noise to all objectives of the meta-data."""
    rng = np.random.default_rng(seed)
    noisy_meta_data: Dict[Union[str, int], List[Evaluation]] = copy.deepcopy(meta_data)

    noise_scale_dict: dict[str, float] = {}
    if isinstance(noise_scale, float):
        noise_scale_dict["loss"] = noise_scale
    else:
        noise_scale_dict.update(noise_scale)

    for task_id in noisy_meta_data.keys():
        for i in range(len(noisy_meta_data[task_id])):
            for k in noisy_meta_data[task_id][i].objectives.keys():
                # skip not recorded objectives
                if noisy_meta_data[task_id][i].objectives[k] is None:
                    continue

                noise: float = noise_scale_dict[k] * rng.standard_normal()
                if noise_is_relative:
                    noise *= noisy_meta_data[task_id][i].objectives[k]  # type: ignore

                noisy_meta_data[task_id][i].objectives[k] += noise  # type: ignore

    return noisy_meta_data


def get_module_string(func: Callable) -> str:
    """Return module string of the file that contains the given function."""
    # Ensure that if __file__ returns an absolute path, e.g. during PyCharm debugging,
    # we use the relative one for constructing the module string
    relative_module_path = (
        f"scamlgp{os.sep}benchmarking"
        + inspect.getmodule(func).__file__.split(f"scamlgp{os.sep}benchmarking")[-1]
    )
    # The :-3 removes the .py file ending
    return relative_module_path.replace(os.sep, ".")[:-3]


def _raise_on_missing_or_invalid_experiments_in_module(module: str):
    """Raise `ValueError` in case the given module is missing an `EXPERIMENTS` global
    dictionary that has only `Experiment` instances as values.
    """
    experiments = get_experiments_config_from_module(module)

    if not isinstance(experiments, dict):
        raise ValueError(
            f"Module {module}.EXPERIMENTS needs to be a dictionary but is "
            + f"{type(experiments)}."
        )

    non_experiment_instances = {
        name: type(config)
        for name, config in experiments.items()
        if not isinstance(config, Experiment)
    }
    if non_experiment_instances:
        raise ValueError(
            f"Module {module}.EXPERIMENTS needs to contain exclusively "
            + f"Experiment instances but contains: {non_experiment_instances}"
        )


def _results_path(func_module: str, name: str, config_hash: str) -> Path:
    return (
        REPO_ROOT
        / Path(*func_module.split(".")[:-1])
        / "results"
        / f"{name}_{config_hash}"
    )


def load_results_from_disk(
    configurations: Dict[str, Experiment], func_module_string: str
):
    all_results = {}
    for name, config in configurations.items():
        config_hash = hash_experiment_config(config)
        local_results_path = _results_path(
            func_module=func_module_string, name=name, config_hash=config_hash
        )

        info_file_path = local_results_path / "info.json"
        if not info_file_path.exists():
            print(f"Unable to load results from {local_results_path}")
        else:
            with open(info_file_path, "r", encoding="UTF-8") as fh:
                all_results[name] = json.load(fh)
                all_results[name]["studies"] = []

            results_files = glob.glob(str(local_results_path / "*.json"))
            for results_file in results_files:
                if "info.json" in results_file:
                    continue
                with open(results_file, "r", encoding="UTF-8") as fh:
                    results = json.load(fh)
                # NOTE: No checks for consistency of the environment and config here
                all_results[name]["studies"].extend(results["studies"])
    return all_results


def with_experiment_cli_and_data_loading(func):
    def wrapper_with_experiment_cli_and_data_loading(
        available_configs: Dict[str, Experiment]
    ):
        func_module_string = get_module_string(func)
        _raise_on_missing_or_invalid_experiments_in_module(func_module_string)

        parser = argparse.ArgumentParser(
            f"Benchmark experiment CLI ({func_module_string})"
        )
        subparsers = parser.add_subparsers(
            title="modes",
            dest="mode",
            help="Submitting jobs, downloading or visualizing results",
            required=True,
        )

        submit_parser = subparsers.add_parser(
            "submit", help="Submit configurations for evaluation"
        )
        submit_parser.add_argument(
            "configurations",
            nargs="+",
            choices=["all"] + list(available_configs.keys()),
            help="The configuration factory names to submit for (re-)computation. "
            + "This can also be 'all' to submit all available configurations.",
        )
        submit_parser.add_argument(
            "--hpobench",
            type=str,
            required=False,
            default=None,
            help="Directory of the HPOBench tabular data. (Only required when using "
            + "--target local with the HPOBenchTabular benchmark.)",
        )
        submit_parser.add_argument(
            "--fcnet",
            type=str,
            required=False,
            default=None,
            help="Directory of the FCNet tabular data. (Only required when using "
            + "--target local with the FCNetFixedFidelityTabularBenchmark benchmark.)",
        )
        submit_parser.add_argument(
            "--parallel-studies",
            type=int,
            required=False,
            default=None,
            help="Override the number of studies to evaluate in parallel within a "
            + "configuration in a single compute instance. If not specified the "
            + "default is no parallelization unless the PARALLEL compute "
            + "cluster is used, in which case 64 studies are ran in parallel.",
        )

        visualize_parser = subparsers.add_parser("visualize", help="Visualize results")
        visualize_parser.add_argument(
            "configurations",
            nargs="+",
            choices=["all"] + list(available_configs.keys()),
            help="Visualize all or only a subset of the results.",
        )

        hash_parser = subparsers.add_parser("hash", help="Visualize results")
        hash_parser.add_argument(
            "configurations",
            nargs="+",
            choices=["all"] + list(available_configs.keys()),
            help="Compute the fingerprint of all or a subset of the configurations.",
        )

        args = parser.parse_args()

        if "all" in args.configurations:
            selected_configs = available_configs
        else:
            selected_configs = {n: available_configs[n] for n in args.configurations}
        print("Selected configurations:", ", ".join(selected_configs.keys()))

        if args.mode == "submit":
            for name, config in selected_configs.items():
                max_workers = (
                    (min(64, os.cpu_count()) if config.compute == "PARALLEL" else 1)
                    if args.parallel_studies is None
                    else args.parallel_studies
                )
                print(f"Starting {name} locally, using max {max_workers} processes")
                submit_local_job(
                    config=config,
                    experiment_key=name,
                    experiment_module=func_module_string,
                    hpobench_path=args.hpobench,
                    fcnet_path=args.fcnet,
                    max_workers=max_workers,
                )

        elif args.mode == "hash":
            for name, config in selected_configs.items():
                print(hash_experiment_config(config), name)

        else:
            results_from_disk = load_results_from_disk(
                configurations=selected_configs, func_module_string=func_module_string
            )
            return func(results_from_disk)

    return wrapper_with_experiment_cli_and_data_loading
