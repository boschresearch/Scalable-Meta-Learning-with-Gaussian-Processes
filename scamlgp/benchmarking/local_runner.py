# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import concurrent.futures
import json
import logging
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import pkg_resources
import torch
from blackboxopt.base import MultiObjectiveOptimizer, SingleObjectiveOptimizer

from scamlgp.benchmarking.bbo_helper import run_with_bbo
from scamlgp.benchmarking.benchmarks.base import Base as BenchmarkBase
from scamlgp.benchmarking.experiment_config_utils import (
    Experiment,
    hash_experiment_config,
    parse_experiment_config,
)
from scamlgp.benchmarking.noise.base import NoiseBase
from scamlgp.benchmarking.noise.benchmark import NoisyBenchmark

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()


def run_study(
    optimizer_cls: Union[Type[SingleObjectiveOptimizer], Type[MultiObjectiveOptimizer]],
    optimizer_kwargs: Dict[str, Any],
    benchmark_cls: Type[BenchmarkBase],
    benchmark_kwargs: Dict[str, Any],
    max_evaluations: int,
    study_seed: int,
    noise_spec: Optional[NoiseBase] = None,
):
    """Run a single study with a given optimizer and benchmark.

    Args:
        optimizer_cls: The `blackboxopt` optimizer class to use.
        optimizer_kwargs: Specific keyword arguments for the optimizer initialization.
        benchmark_cls: The benchmark class to use.
        benchmark_kwargs: Specific keyword arguments for the benchmark initialization.
        max_evaluations: The maximum number of evaluations to run.
        study_seed: The seed for the study.
        noise_spec: The noise specification for the benchmark.
    """
    if noise_spec is not None:
        noise_spec.rng = np.random.default_rng(study_seed)
        benchmark = NoisyBenchmark(
            benchmark_cls(**benchmark_kwargs, seed=study_seed), noise_spec
        )
    else:
        benchmark = benchmark_cls(**benchmark_kwargs, seed=study_seed)

    evaluations = run_with_bbo(
        benchmark=benchmark,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs_from_config=optimizer_kwargs,
        max_evaluations=max_evaluations,
        meta_data_seed=study_seed,
    )

    if isinstance(benchmark, NoisyBenchmark):
        for ev in evaluations:
            noise_free_eval = benchmark.noise_free_benchmark(ev.get_specification())
            objectives = {
                **{f"{n} (noisy)": v for n, v in ev.objectives.items()},
                **{
                    f"{n} (noise free)": v
                    for n, v in noise_free_eval.objectives.items()
                },
            }
            ev.objectives = objectives

    return {
        "optimum": getattr(benchmark, "optimum", None),
        "objectives": [o.__dict__ for o in benchmark.objectives],
        "evaluations": [e.__dict__ for e in evaluations],
        "seed": study_seed,
    }


def main(
    config: Experiment,
    experiment_module: str,
    experiment_key: str,
    max_workers: int,
    hpobench_path: Optional[str] = None,
    fcnet_path: Optional[str] = None,
):
    """Run a given experiment configuration and save the results to disk.

    Args:
        config: The experiment configuration.
        experiment_module: The Python module of the experiment configuration.
        experiment_key: The key of the experiment configuration for human readability.
        max_workers: The number of workers to use for parallel computation.
        hpobench_path: The path to the HPOBench benchmark data.
        fcnet_path: The path to the FCNet benchmark data.
    """
    logging.getLogger("blackboxopt").setLevel(logging.WARNING)

    if max_workers > 1:
        torch.set_num_threads(1)  # pylint: disable=E1101

    benchmark_kwargs = (
        config.benchmark["kwargs"] if isinstance(config.benchmark, dict) else {}
    )
    if hpobench_path:
        benchmark_kwargs["data_dir"] = hpobench_path
    if fcnet_path:
        benchmark_kwargs["target_task_file"] = str(
            Path(fcnet_path)
            / "fcnet_tabular_benchmarks"
            / benchmark_kwargs["target_task_file"]
        )
        benchmark_kwargs["meta_task_files"] = [
            str(Path(fcnet_path) / "fcnet_tabular_benchmarks" / mtf)
            for mtf in benchmark_kwargs["meta_task_files"]
        ]

    config_hash = hash_experiment_config(config)

    output_dir = (
        REPO_ROOT
        / Path(*experiment_module.split(".")[:-1])
        / "results"
        / f"{experiment_key}_{config_hash}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Record general information about environment & run
    info = dict(
        experiment_config=parse_experiment_config(config.__dict__),
        experiment_module=experiment_module,
        experiment_key=experiment_key,
        environment={
            d.project_name: d.version for d in list(pkg_resources.working_set)
        },
        timestamp=time.time(),
    )
    info_file_path = output_dir / "info.json"
    with open(info_file_path, "w", encoding="UTF-8") as fh:
        json.dump(info, fh)

    # Prep all kwargs aside from study_seed
    _run_study = partial(
        run_study,
        optimizer_cls=(
            config.optimizer
            if not isinstance(config.optimizer, dict)
            else config.optimizer["cls"]
        ),
        optimizer_kwargs=(
            {} if not isinstance(config.optimizer, dict) else config.optimizer["kwargs"]
        ),
        benchmark_cls=(
            config.benchmark["cls"]
            if isinstance(config.benchmark, dict)
            else config.benchmark
        ),
        benchmark_kwargs=benchmark_kwargs,
        max_evaluations=config.n_evaluations,
        noise_spec=(
            config.benchmark.get("noise_spec", None)
            if isinstance(config.benchmark, dict)
            else None
        ),
    )
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers
        # With Py3.11, add: max_tasks_per_child=1
    ) as executor:
        futures = [
            executor.submit(_run_study, study_seed=seed)
            for seed in range(config.n_studies)
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                benchmark_results = future.result()
                study_seed = benchmark_results["seed"]

                results = dict(
                    experiment_config=parse_experiment_config(config.__dict__),
                    experiment_module=experiment_module,
                    experiment_key=experiment_key,
                    timestamp=time.time(),
                    # Plural studies and a singular list for backwards compatibility
                    studies=[benchmark_results],
                )

                output_file_path = (
                    output_dir / f"{experiment_key}_{study_seed}_{config_hash}.json"
                )
                with open(output_file_path, "w", encoding="UTF-8") as fh:
                    json.dump(results, fh)

            except Exception:
                print("Error loading result")
                traceback.print_exc()
