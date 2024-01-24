# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path
from typing import Any, Dict

from blackboxopt import Objective

from scamlgp.benchmarking.benchmarks.hpo_bench_tabular import HPOBenchTabular
from scamlgp.benchmarking.configurations.styles import OPTIMIZER_STYLES
from scamlgp.benchmarking.experiment_config_utils import Experiment
from scamlgp.benchmarking.plotting import grouped_results
from scamlgp.benchmarking.utils import with_experiment_cli_and_data_loading
from scamlgp.optimizer import ScaMLGPBO


@with_experiment_cli_and_data_loading
def main(results: dict):
    fig = grouped_results(
        list(results.values()),
        optimizer_styles=OPTIMIZER_STYLES,
        groups={
            "RF Tabular\n27 Tasks à 64 Points": [
                v for k, v in EXPERIMENTS.items() if k.startswith("RF_T27_P64")
            ],
        },
        robust_statistics=False,
        use_regrets=True,
        objective=Objective("1 - Accuracy", False),
    )
    fig.savefig(
        Path(__file__).parent / f"{Path(__file__).name[:-3]}_benchmark_wise_regrets.pdf"
    )
    fig.savefig(
        Path(__file__).parent / f"{Path(__file__).name[:-3]}_benchmark_wise_regrets.png"
    )
    b = HPOBenchTabular(scenario="rf", n_data_per_task=[])
    print(b.search_space)
    print(len(b.search_space))


_RF_T27_P128_CONFIG = {
    "cls": HPOBenchTabular,
    "kwargs": {"scenario": "rf", "n_data_per_task": [128] * 27},
}

_DEFAULTS: Dict[str, Any] = dict(n_evaluations=60, n_studies=256, compute="PARALLEL")

EXPERIMENTS = {
    "RF_T27_P128_SKALEGP": Experiment(
        benchmark=_RF_T27_P128_CONFIG, optimizer=ScaMLGPBO, **_DEFAULTS
    ),
}

if __name__ == "__main__":
    main(EXPERIMENTS)
