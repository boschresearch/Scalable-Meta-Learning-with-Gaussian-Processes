# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
from blackboxopt import Objective

from scamlgp.benchmarking.benchmarks.pd1 import PD1
from scamlgp.benchmarking.configurations.styles import OPTIMIZER_STYLES
from scamlgp.benchmarking.experiment_config_utils import Experiment
from scamlgp.benchmarking.plotting import grouped_results
from scamlgp.benchmarking.utils import with_experiment_cli_and_data_loading
from scamlgp.optimizer import ScaMLGPBO


@with_experiment_cli_and_data_loading
def main(results: dict):
    plt.rcParams.update({"text.usetex": True})

    fig = grouped_results(
        list(results.values()),
        optimizer_styles=OPTIMIZER_STYLES,
        groups={"PD1\n" + r"$M=22 \quad N_m=128$": EXPERIMENTS.values()},
        robust_statistics=False,
        use_regrets=True,
        objective=Objective("best_valid/error_rate", False),
    )
    fig.savefig(
        Path(__file__).parent / f"{Path(__file__).name[:-3]}_benchmark_wise_regrets.pdf"
    )
    fig.savefig(
        Path(__file__).parent / f"{Path(__file__).name[:-3]}_benchmark_wise_regrets.png"
    )


_PD1_T22_P128_CONFIG = {"cls": PD1, "kwargs": {"n_data_per_task": [128] * 22}}

_DEFAULTS: Dict[str, Any] = dict(n_evaluations=60, n_studies=256, compute="PARALLEL")


EXPERIMENTS = {
    "PD1_T22_P128_SCAMLGP": Experiment(
        benchmark=_PD1_T22_P128_CONFIG, optimizer=ScaMLGPBO, **_DEFAULTS
    ),
}

if __name__ == "__main__":
    main(EXPERIMENTS)
