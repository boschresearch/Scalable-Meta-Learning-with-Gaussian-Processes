# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path
from typing import Any, Dict

from blackboxopt import Objective

from scamlgp.benchmarking.benchmarks.hartmann_3d import Hartmann3D
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
                "Hartmann3\n8 Tasks à 32 Points (σ_noise=0.1)": [
                    v for k, v in EXPERIMENTS.items() if k.startswith("HM3_T8_P32_N01")
                ],
                "Hartmann3\n32 Tasks à 32 Points (σ_noise=0.1)": [
                    v for k, v in EXPERIMENTS.items() if k.startswith("HM3_T32_P32_N01")
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


DEFAULTS_HARTMANN_3D: Dict[str, Any] = dict(
    n_evaluations=40, n_studies=128, compute="PARALLEL"
)

HM3_CONFIG = {
    "cls": Hartmann3D,
    "kwargs": {"n_data_per_task": []},
    "noise_spec": HomoscedasticGaussianNoise(noise_std={"loss": 0.1}),
}
HM3_T8_P32_CONFIG = {
    "cls": Hartmann3D,
    "kwargs": {"n_data_per_task": [32] * 8},
    "noise_spec": HomoscedasticGaussianNoise(noise_std={"loss": 0.1}),
}
HM3_T32_P32_CONFIG = {
    "cls": Hartmann3D,
    "kwargs": {"n_data_per_task": [32] * 32},
    "noise_spec": HomoscedasticGaussianNoise(noise_std={"loss": 0.1}),
}
HM3_T32_P256_CONFIG = {
    "cls": Hartmann3D,
    "kwargs": {"n_data_per_task": [256] * 32},
    "noise_spec": HomoscedasticGaussianNoise(noise_std={"loss": 0.1}),
}


EXPERIMENTS = {
    "HM3_T8_P32_N01_SCAMLGP": Experiment(
        benchmark=HM3_T8_P32_CONFIG, optimizer=ScaMLGPBO, **DEFAULTS_HARTMANN_3D
    ),
    "HM3_T32_P32_N01_SCAMLGP": Experiment(
        benchmark=HM3_T32_P32_CONFIG, optimizer=ScaMLGPBO, **DEFAULTS_HARTMANN_3D
    ),
}

if __name__ == "__main__":
    main(EXPERIMENTS)
