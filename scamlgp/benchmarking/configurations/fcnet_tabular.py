# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
from blackboxopt import Objective

from scamlgp.benchmarking.benchmarks.fcnet_fixed_fidelity_tabular import (
    FCNetFixedFidelityTabularBenchmark,
)
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
        groups={
            "Slice\n"
            + r"$M=3 \quad N_m=256$": [
                v for k, v in EXPERIMENTS.items() if k.startswith("SLICE_")
            ],
            "Protein\n"
            + r"$M=3 \quad N_m=256$": [
                v for k, v in EXPERIMENTS.items() if k.startswith("PROTEIN_")
            ],
            "Naval\n"
            + r"$M=3 \quad N_m=256$": [
                v for k, v in EXPERIMENTS.items() if k.startswith("NAVAL_")
            ],
            "Parkinson's\n"
            + r"$M=3 \quad N_m=256$": [
                v for k, v in EXPERIMENTS.items() if k.startswith("PARKIN_")
            ],
        },
        robust_statistics=False,
        use_regrets=True,
        objective=Objective("valid_loss", False),
        rel_fig_width=0.8,
        x_limits=[(1, 40), (1, 40), (1, 40), (1, 40)],
        y_limits=[(1e-4, 1e-2), (1e-3, 1e-1), (1e-5, 1e-2), (2e-3, 1e-1)],
    )
    fig.savefig(
        Path(__file__).parent / f"{Path(__file__).name[:-3]}_benchmark_wise_regrets.pdf"
    )
    fig.savefig(
        Path(__file__).parent / f"{Path(__file__).name[:-3]}_benchmark_wise_regrets.png"
    )


_T3_P256_KWARGS = {
    "n_data_per_task": [256] * 3,
    # Ensure we don't expose the categorical values for optimization but use the
    # (close to) optimal values instead, always.
    "fix_search_space": {
        "activation_fn_1": "relu",
        "activation_fn_2": "relu",
        "lr_schedule": "cosine",
    },
}

_SLICE_T3_P256_CONFIG = {
    "cls": FCNetFixedFidelityTabularBenchmark,
    "kwargs": {
        "target_task_file": "fcnet_slice_localization_data.hdf5",
        "meta_task_files": [
            "fcnet_naval_propulsion_data.hdf5",
            "fcnet_parkinsons_telemonitoring_data.hdf5",
            "fcnet_protein_structure_data.hdf5",
        ],
        **_T3_P256_KWARGS,
    },
}
_NAVAL_T3_P256_CONFIG = {
    "cls": FCNetFixedFidelityTabularBenchmark,
    "kwargs": {
        "target_task_file": "fcnet_naval_propulsion_data.hdf5",
        "meta_task_files": [
            "fcnet_parkinsons_telemonitoring_data.hdf5",
            "fcnet_protein_structure_data.hdf5",
            "fcnet_slice_localization_data.hdf5",
        ],
        **_T3_P256_KWARGS,
    },
}
_PARKIN_T3_P256_CONFIG = {
    "cls": FCNetFixedFidelityTabularBenchmark,
    "kwargs": {
        "target_task_file": "fcnet_parkinsons_telemonitoring_data.hdf5",
        "meta_task_files": [
            "fcnet_naval_propulsion_data.hdf5",
            "fcnet_protein_structure_data.hdf5",
            "fcnet_slice_localization_data.hdf5",
        ],
        **_T3_P256_KWARGS,
    },
}
_PROTEIN_T3_P256_CONFIG = {
    "cls": FCNetFixedFidelityTabularBenchmark,
    "kwargs": {
        "target_task_file": "fcnet_protein_structure_data.hdf5",
        "meta_task_files": [
            "fcnet_naval_propulsion_data.hdf5",
            "fcnet_parkinsons_telemonitoring_data.hdf5",
            "fcnet_slice_localization_data.hdf5",
        ],
        **_T3_P256_KWARGS,
    },
}

_DEFAULTS: Dict[str, Any] = dict(n_evaluations=80, n_studies=128, compute="PARALLEL")


EXPERIMENTS = {
    "SLICE_T3_P256_SCAMLGP": Experiment(
        benchmark=_SLICE_T3_P256_CONFIG, optimizer=ScaMLGPBO, **_DEFAULTS
    ),
    "PROTEIN_T3_P256_SCAMLGP": Experiment(
        benchmark=_PROTEIN_T3_P256_CONFIG, optimizer=ScaMLGPBO, **_DEFAULTS
    ),
    "PARKIN_T3_P256_SCAMLGP": Experiment(
        benchmark=_PARKIN_T3_P256_CONFIG, optimizer=ScaMLGPBO, **_DEFAULTS
    ),
    "NAVAL_T3_P256_SCAMLGP": Experiment(
        benchmark=_NAVAL_T3_P256_CONFIG, optimizer=ScaMLGPBO, **_DEFAULTS
    ),
}

if __name__ == "__main__":
    main(EXPERIMENTS)
