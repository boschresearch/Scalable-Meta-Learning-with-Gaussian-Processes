# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path
from typing import Any, Dict

from blackboxopt import Objective

from scamlgp.benchmarking.benchmarks.hartmann_6d import Hartmann6D
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
                "Hartmann6D\n8 Tasks à 128 Points": [
                    v
                    for k, v in EXPERIMENTS.items()
                    if k.startswith("HARTMANN_6D_T8_P128_N01")
                ],
                "Hartmann6D\n32 Tasks à 128 Points": [
                    v
                    for k, v in EXPERIMENTS.items()
                    if k.startswith("HARTMANN_6D_T32_P128_N01")
                ],
            },
            robust_statistics=robust_statistics,
            use_regrets=True,
            use_benchmark_optimum=True,
            objective=Objective("loss", greater_is_better=False),
        )
        stats_label = "median_25quant75" if robust_statistics else "mean_sem"
        fig.savefig(
            Path(__file__).parent
            / f"{Path(__file__).name[:-3]}_benchmark_wise_regrets_{stats_label}.pdf"
        )


DEFAULTS_HARTMANN_6D: Dict[str, Any] = dict(
    n_evaluations=80, n_studies=128, compute="PARALLEL"
)

HM6_CONFIG = {
    "cls": Hartmann6D,
    "kwargs": {"n_data_per_task": []},
    "noise_spec": HomoscedasticGaussianNoise(noise_std={"loss": 0.1}),
}
HM6_T8_P128_CONFIG = {
    "cls": Hartmann6D,
    "kwargs": {"n_data_per_task": [128] * 8},
    "noise_spec": HomoscedasticGaussianNoise(noise_std={"loss": 0.1}),
}
HM6_T32_P128_CONFIG = {
    "cls": Hartmann6D,
    "kwargs": {"n_data_per_task": [128] * 32},
    "noise_spec": HomoscedasticGaussianNoise(noise_std={"loss": 0.1}),
}

EXPERIMENTS = {
    "HARTMANN_6D_T8_P128_N01_SCAMLGP": Experiment(
        benchmark=HM6_T8_P128_CONFIG, optimizer=ScaMLGPBO, **DEFAULTS_HARTMANN_6D
    ),
    "HARTMANN_6D_T32_P128_N01_SCAMLGP": Experiment(
        benchmark=HM6_T32_P128_CONFIG, optimizer=ScaMLGPBO, **DEFAULTS_HARTMANN_6D
    ),
}

if __name__ == "__main__":
    main(EXPERIMENTS)
