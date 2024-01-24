# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
from blackboxopt import Objective

from scamlgp.benchmarking.benchmarks.hartmann_6d import Hartmann6D
from scamlgp.benchmarking.configurations.styles import OPTIMIZER_STYLES
from scamlgp.benchmarking.noise.homoscedastic import HomoscedasticGaussianNoise
from scamlgp.benchmarking.plotting import grouped_results
from scamlgp.benchmarking.utils import Experiment, with_experiment_cli_and_data_loading
from scamlgp.optimizer import ScaMLGPBO

NUM_META_TASKS = 8
NUM_POINTS_PER_TASK = [16, 32, 64, 128, 256, 512]


@with_experiment_cli_and_data_loading
def main(results: dict):
    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

    groups = {
        f"Hartmann 6D\n$M={NUM_META_TASKS} \quad N_m={num_points}$": [
            v
            for k, v in EXPERIMENTS.items()
            if k.startswith(f"HARTMANN_6D_T{NUM_META_TASKS}_P{num_points}_N01")
        ]
        for num_points in NUM_POINTS_PER_TASK
    }

    for robust_statistics in [True, False]:
        fig = grouped_results(
            list(results.values()),
            optimizer_styles=OPTIMIZER_STYLES,
            groups=groups,
            robust_statistics=robust_statistics,
            use_regrets=True,
            use_benchmark_optimum=True,
            objective=Objective("loss", greater_is_better=False),
            sharey="row",
            x_limits=[(1, _DEFAULTS["n_evaluations"])] * len(groups),
        )
        stats_label = "median_25quant75" if robust_statistics else "mean_sem"
        fig.savefig(
            Path(__file__).parent
            / f"{Path(__file__).name[:-3]}_benchmark_wise_regrets_{stats_label}.pdf"
        )


_DEFAULTS: Dict[str, Any] = dict(n_evaluations=80, n_studies=128, compute="PARALLEL")

OPTIMIZERS = {"SCAMLGP": ScaMLGPBO}

EXPERIMENTS = {}
for num_points in NUM_POINTS_PER_TASK:
    benchmark_config = {
        "cls": Hartmann6D,
        "kwargs": {"n_data_per_task": [num_points] * NUM_META_TASKS},
        "noise_spec": HomoscedasticGaussianNoise(noise_std={"loss": 0.1}),
    }
    for optimizer_name, optimizer in OPTIMIZERS.items():
        EXPERIMENTS[
            f"HARTMANN_6D_T{NUM_META_TASKS}_P{num_points}_N01_{optimizer_name}"
        ] = Experiment(benchmark=benchmark_config, optimizer=optimizer, **_DEFAULTS)


if __name__ == "__main__":
    main(EXPERIMENTS)
