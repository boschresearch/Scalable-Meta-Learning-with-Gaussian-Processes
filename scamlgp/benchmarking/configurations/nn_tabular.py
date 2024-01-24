# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from blackboxopt import Evaluation, Objective
from scipy.stats import sem

from scamlgp.benchmarking.benchmarks.hpo_bench_tabular import HPOBenchTabular
from scamlgp.benchmarking.configurations.styles import OPTIMIZER_STYLES
from scamlgp.benchmarking.experiment_config_utils import Experiment
from scamlgp.benchmarking.plotting import compute_regrets, grouped_results
from scamlgp.benchmarking.utils import with_experiment_cli_and_data_loading
from scamlgp.optimizer import ScaMLGPBO


def _regret_stats(studies: list) -> Tuple[np.ndarray, np.ndarray]:
    regrets = [
        compute_regrets(
            objective=(
                Objective(
                    name=s["objectives"][0]["name"] + " (noise free)",
                    greater_is_better=s["objectives"][0]["greater_is_better"],
                )
                if s["objectives"][0]["name"] + " (noise free)"
                in s["evaluations"][0]["objectives"].keys()
                else Objective(**s["objectives"][0])
            ),
            optimum=s["optimum"],
            objective_values=[Evaluation(**e).objectives for e in s["evaluations"]],
        )
        for s in studies
    ]

    center_statistic = np.mean(regrets, axis=0).squeeze()
    regrets_sem = sem(regrets, axis=0).squeeze()

    return center_statistic, regrets_sem


def _print_markdown_table(
    results: dict, configs: List[Experiment], iterations: List[int]
):
    runs_data = list(results.values())

    ii = [i - 1 for i in iterations]

    run_configs = [Experiment(**run["experiment_config"]) for run in runs_data]

    print("| optimizer | " + " | ".join([str(i) for i in iterations]), "|")
    print("|", " | ".join(["---" for _ in range(len(iterations) + 1)]), "|")

    stats = []
    # group contains a tuple of configs
    for config in configs:
        try:
            i_config = run_configs.index(config)
        except ValueError:
            print(
                "Unable to find configuration in available results, skipping",
                json.dumps(config.__dict__, indent=2),
            )
            continue
        data = runs_data[i_config]

        optimizer_style = (
            OPTIMIZER_STYLES[config.optimizer["cls"]]
            if isinstance(config.optimizer, dict)
            else OPTIMIZER_STYLES[config.optimizer]
        )
        regrets_mean, regrets_sem = _regret_stats(data["studies"])
        stats.append((optimizer_style["label"], regrets_mean, regrets_sem))

    stats = sorted(stats, key=lambda x: x[1][-1])

    for label, regrets_mean, regrets_sem in stats:
        print(
            "|",
            label,
            "|",
            "|".join(
                [
                    f"{m:0.2E} +/- {e:0.2E}"
                    for m, e in zip(regrets_mean[ii], regrets_sem[ii])
                ]
            ),
            "|",
        )


@with_experiment_cli_and_data_loading
def main(results: dict):
    _print_markdown_table(
        results, configs=EXPERIMENTS.values(), iterations=[10, 20, 30, 40, 50, 60]
    )

    fig = grouped_results(
        list(results.values()),
        optimizer_styles=OPTIMIZER_STYLES,
        groups={
            "NN Tabular\n7 Tasks à 64 Points": [
                v for k, v in EXPERIMENTS.items() if k.startswith("NN_T7_P128")
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

    b = HPOBenchTabular(scenario="nn", n_data_per_task=[])
    print(b.search_space)
    print(len(b.search_space))


_NN_T7_P128_CONFIG = {
    "cls": HPOBenchTabular,
    "kwargs": {"scenario": "nn", "n_data_per_task": [128] * 7},
}

_DEFAULTS: Dict[str, Any] = dict(n_evaluations=60, n_studies=256, compute="PARALLEL")

EXPERIMENTS = {
    "NN_T7_P128_SCAMLGP": Experiment(
        benchmark=_NN_T7_P128_CONFIG, optimizer=ScaMLGPBO, **_DEFAULTS
    ),
}

if __name__ == "__main__":
    main(EXPERIMENTS)
