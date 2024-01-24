# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path

import matplotlib.pyplot as plt
from blackboxopt import Objective

from scamlgp.benchmarking.configurations.fcnet_tabular import (
    EXPERIMENTS as EXPERIMENTS_FCNET,
)
from scamlgp.benchmarking.configurations.lr_tabular import EXPERIMENTS as EXPERIMENTS_LR
from scamlgp.benchmarking.configurations.nn_tabular import EXPERIMENTS as EXPERIMENTS_NN
from scamlgp.benchmarking.configurations.rf_tabular import EXPERIMENTS as EXPERIMENTS_RF
from scamlgp.benchmarking.configurations.styles import OPTIMIZER_STYLES
from scamlgp.benchmarking.configurations.svm_tabular import (
    EXPERIMENTS as EXPERIMENTS_SVM,
)
from scamlgp.benchmarking.configurations.xgb_tabular import (
    EXPERIMENTS as EXPERIMENTS_XGB,
)
from scamlgp.benchmarking.plotting import grouped_results
from scamlgp.benchmarking.utils import with_experiment_cli_and_data_loading


@with_experiment_cli_and_data_loading
def main(results: dict):
    plt.rcParams.update({"text.usetex": True})

    fig = grouped_results(
        list(results.values()),
        optimizer_styles=OPTIMIZER_STYLES,
        groups={
            "SVM\n"
            + r"$M=28 \quad N_m=64$": [
                v for k, v in EXPERIMENTS.items() if k.startswith("SVM_T28_P64")
            ],
            "MLP\n"
            + r"$M=7 \quad N_m=128$": [
                v for k, v in EXPERIMENTS.items() if k.startswith("NN_T7_P128")
            ],
            "XGB\n"
            + r"$M=19 \quad N_m=128$": [
                v for k, v in EXPERIMENTS.items() if k.startswith("XGB_T19_P128")
            ],
            "RF\n"
            + r"$M=27 \quad N_m=128$": [
                v for k, v in EXPERIMENTS.items() if k.startswith("RF_T27_P128")
            ],
            "LR\n"
            + r"$M=28 \quad N_m=64$": [
                v for k, v in EXPERIMENTS.items() if k.startswith("LR_T28_P64")
            ],
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
        objective=[
            Objective("1 - Accuracy", False),
            Objective("1 - Accuracy", False),
            Objective("1 - Accuracy", False),
            Objective("1 - Accuracy", False),
            Objective("1 - Accuracy", False),
            Objective("valid_loss", False),
            Objective("valid_loss", False),
            Objective("valid_loss", False),
            Objective("valid_loss", False),
        ],
        n_rows=3,
        n_cols=3,
        h_pad=1.2,
        fig_height=5.0,
        x_limits=[(1, 60)] * 9,
        y_limits=[
            (1e-3, 1e-1),
            (2e-3, 5e-2),
            (4e-4, 1e-2),
            (1e-4, 5e-2),
            (4e-4, 1e-2),
            (1e-4, 1e-2),
            (1e-3, 1e-1),
            (1e-5, 1e-2),
            (2e-3, 1e-1),
        ],
    )
    fig.savefig(
        Path(__file__).parent / f"{Path(__file__).name[:-3]}_benchmark_wise_regrets.pdf"
    )


EXPERIMENTS = {
    **EXPERIMENTS_LR,
    **EXPERIMENTS_NN,
    **EXPERIMENTS_RF,
    **EXPERIMENTS_SVM,
    **EXPERIMENTS_XGB,
    **EXPERIMENTS_FCNET,
}


if __name__ == "__main__":
    main(EXPERIMENTS)
