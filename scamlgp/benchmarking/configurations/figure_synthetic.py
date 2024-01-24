# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path

import matplotlib.pyplot as plt
from blackboxopt import Objective

from scamlgp.benchmarking.configurations.branin import EXPERIMENTS as EXPERIMENTS_BRANIN
from scamlgp.benchmarking.configurations.hartmann3 import (
    EXPERIMENTS as EXPERIMENTS_HARTMANN_3D,
)
from scamlgp.benchmarking.configurations.hartmann6 import (
    EXPERIMENTS as EXPERIMENTS_HARTMANN_6D,
)
from scamlgp.benchmarking.configurations.styles import OPTIMIZER_STYLES
from scamlgp.benchmarking.plotting import grouped_results
from scamlgp.benchmarking.utils import with_experiment_cli_and_data_loading


@with_experiment_cli_and_data_loading
def main(results: dict):
    plt.rcParams.update({"text.usetex": True})

    for robust_statistics in [True, False]:
        fig = grouped_results(
            list(results.values()),
            optimizer_styles=OPTIMIZER_STYLES,
            groups={
                "Branin\n"
                + r"$M=8 \quad N_m=32$": [
                    v
                    for k, v in EXPERIMENTS.items()
                    if k.startswith("BRANIN_T8_P32_N1")
                ],
                "Hartmann 3D\n"
                + r"$M=8 \quad N_m=32$": [
                    v for k, v in EXPERIMENTS.items() if k.startswith("HM3_T8_P32_N01")
                ],
                "Hartmann 6D\n"
                + r"$M=8 \quad N_m=128$": [
                    v
                    for k, v in EXPERIMENTS.items()
                    if k.startswith("HARTMANN_6D_T8_P128_N01")
                ],
                r"$M=32 \quad N_m=32$": [
                    v
                    for k, v in EXPERIMENTS.items()
                    if k.startswith("BRANIN_T32_P32_N1")
                ],
                # This key contains a space to allow for the same text to be rendered
                # effectively, while still respecting that dictionary keys need to be
                # unique
                r"$M=32 \quad N_m=32$ ": [
                    v for k, v in EXPERIMENTS.items() if k.startswith("HM3_T32_P32_N01")
                ],
                r"$M=32 \quad N_m=128$": [
                    v
                    for k, v in EXPERIMENTS.items()
                    if k.startswith("HARTMANN_6D_T32_P128_N01")
                ],
            },
            use_regrets=True,
            robust_statistics=robust_statistics,
            use_benchmark_optimum=True,
            objective=Objective("loss", greater_is_better=False),
            x_limits=[
                (1, 40),
                (1, 40),
                (1, 80),
                (1, 40),
                (1, 40),
                (1, 80),
            ],
            y_limits=[
                (2e-2, 10.0),
                (5e-3, 1.0),
                (3e-2, 1.0),
                (2e-2, 10.0),
                (5e-3, 1.0),
                (3e-2, 1.0),
            ],
        )
        stats_label = "median_25quant75" if robust_statistics else "mean_sem"
        fig.savefig(
            Path(__file__).parent
            / f"{Path(__file__).name[:-3]}_benchmark_wise_regrets_{stats_label}.pdf"
        )


EXPERIMENTS = {
    **EXPERIMENTS_BRANIN,
    **EXPERIMENTS_HARTMANN_3D,
    **EXPERIMENTS_HARTMANN_6D,
}


if __name__ == "__main__":
    main(EXPERIMENTS)
