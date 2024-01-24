# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path
from typing import Any, Dict

from blackboxopt import Objective

from scamlgp.benchmarking.benchmarks.branin import Branin
from scamlgp.benchmarking.configurations.styles import OPTIMIZER_STYLES
from scamlgp.benchmarking.noise.homoscedastic import HomoscedasticGaussianNoise
from scamlgp.benchmarking.plotting import grouped_results
from scamlgp.benchmarking.utils import Experiment, with_experiment_cli_and_data_loading
from scamlgp.optimizer import ScaMLGPBO


@with_experiment_cli_and_data_loading
def main(results: dict):
    for robust_statistics in [True, False]:
        fig = grouped_results(
            list(results.values()),
            optimizer_styles=OPTIMIZER_STYLES,
            groups={
                "Branin\n8 Tasks à 32 Points (σ_noise=1.0)": [
                    v
                    for k, v in EXPERIMENTS.items()
                    if k.startswith("BRANIN_T8_P32_N1")
                ],
                "Branin\n32 Tasks à 32 Points (σ_noise=1.0)": [
                    v
                    for k, v in EXPERIMENTS.items()
                    if k.startswith("BRANIN_T32_P32_N1")
                ],
            },
            use_regrets=True,
            robust_statistics=robust_statistics,
            use_benchmark_optimum=True,
            objective=Objective("loss", greater_is_better=False),
        )
        stats_label = "median_25quant75" if robust_statistics else "mean_sem"
        fig.savefig(
            Path(__file__).parent
            / f"{Path(__file__).name[:-3]}_benchmark_wise_regrets_{stats_label}.pdf"
        )


DEFAULTS_BRANIN: Dict[str, Any] = dict(
    n_evaluations=40, n_studies=128, compute="PARALLEL"
)

BRANIN_CONFIG = {
    "cls": Branin,
    "kwargs": {"n_data_per_task": []},
    "noise_spec": HomoscedasticGaussianNoise(noise_std={"loss": 1.0}),
}
BRANIN_T8_P32_CONFIG = {
    "cls": Branin,
    "kwargs": {"n_data_per_task": [32] * 8},
    "noise_spec": HomoscedasticGaussianNoise(noise_std={"loss": 1.0}),
}
BRANIN_T32_P32_CONFIG = {
    "cls": Branin,
    "kwargs": {"n_data_per_task": [32] * 32},
    "noise_spec": HomoscedasticGaussianNoise(noise_std={"loss": 1.0}),
}


EXPERIMENTS = {
    "BRANIN_T8_P32_N1_SCAMLGP": Experiment(
        benchmark=BRANIN_T8_P32_CONFIG, optimizer=ScaMLGPBO, **DEFAULTS_BRANIN
    ),
    "BRANIN_T32_P32_N1_SCAMLGP": Experiment(
        benchmark=BRANIN_T32_P32_CONFIG, optimizer=ScaMLGPBO, **DEFAULTS_BRANIN
    ),
}

if __name__ == "__main__":
    main(EXPERIMENTS)
