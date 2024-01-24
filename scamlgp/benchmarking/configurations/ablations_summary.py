# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path

import matplotlib.pyplot as plt

from scamlgp.benchmarking.configurations import (
    branin_ablation_num_meta_tasks,
    branin_ablation_num_points_per_task,
    hartmann6_ablation_num_meta_tasks,
    hartmann6_ablation_num_points_per_task,
)
from scamlgp.benchmarking.configurations.styles import OPTIMIZER_STYLES
from scamlgp.benchmarking.plotting import plot_meta_data_summary_comparison
from scamlgp.benchmarking.utils import with_experiment_cli_and_data_loading


def _set_xticks(ax: plt.Axes, major_ticks: list, minor_ticks: list, set_label: bool):
    ax.set_xticks(major_ticks, minor=False)
    ax.set_xticks(minor_ticks, minor=True)
    if set_label:
        ax.set_xticklabels([str(t) for t in major_ticks], minor=False)
        ax.set_xticklabels(["" for _ in minor_ticks], minor=True)


@with_experiment_cli_and_data_loading
def main(results: dict):
    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

    rel_plot_width = 1.0
    fig, axs = plt.subplots(
        2, 2, figsize=(6.75 * rel_plot_width, 4), sharey="row", sharex="col"
    )

    plot_meta_data_summary_comparison(
        results={k: v for k, v in results.items() if k.lower().startswith("branin")},
        num_points_per_task=branin_ablation_num_meta_tasks.NUM_POINTS_PER_TASK,
        styles=OPTIMIZER_STYLES,
        ax=axs[0, 0],
    )
    axs[0, 0].set_title("Branin")
    axs[0, 0].set_ylabel("Cum. regret")
    plot_meta_data_summary_comparison(
        results={k: v for k, v in results.items() if k.lower().startswith("branin")},
        num_meta_tasks=branin_ablation_num_points_per_task.NUM_META_TASKS,
        styles=OPTIMIZER_STYLES,
        ax=axs[0, 1],
    )
    axs[0, 1].set_title("Branin")

    plot_meta_data_summary_comparison(
        results={
            k: v for k, v in results.items() if k.lower().startswith("hartmann_6d")
        },
        num_points_per_task=hartmann6_ablation_num_meta_tasks.NUM_POINTS_PER_TASK,
        styles=OPTIMIZER_STYLES,
        ax=axs[1, 0],
    )
    axs[1, 0].set_title("Hartmann 6D")
    axs[1, 0].set_xlabel("Num. meta-tasks ($M$)")
    axs[1, 0].set_ylabel("Cum. regret")
    plot_meta_data_summary_comparison(
        results={
            k: v for k, v in results.items() if k.lower().startswith("hartmann_6d")
        },
        num_meta_tasks=hartmann6_ablation_num_points_per_task.NUM_META_TASKS,
        styles=OPTIMIZER_STYLES,
        ax=axs[1, 1],
    )
    axs[1, 1].set_title("Hartmann 6D")
    axs[1, 1].set_xlabel("Num. obs. per task ($N_m$)")

    _set_xticks(
        axs[0, 0],
        [4, 16, 64],
        [2, 8, 32],
        set_label=False,
    )
    _set_xticks(
        axs[0, 1],
        [16, 64, 256],
        [8, 32, 512],
        set_label=False,
    )
    _set_xticks(
        axs[1, 0],
        [4, 16, 64],
        [2, 8, 32],
        set_label=True,
    )
    _set_xticks(
        axs[1, 1],
        [16, 64, 256],
        [8, 32, 512],
        set_label=True,
    )

    handles, labels = [], []
    for ax in axs.flatten():
        handles, labels = ax.get_legend_handles_labels()
        handles.extend(handles)
        labels.extend(labels)
    by_label = dict(zip(labels, handles))
    # Manual order specification to sort them consistently with the other visualizations
    order = [4, 2, 8, 5, 3, 7, 1, 6, 0]
    fig.legend(
        [list(by_label.values())[i] for i in order],
        [list(by_label.keys())[i] for i in order],
        loc="center right",
        ncol=1,
        labelspacing=1.5,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.8 - 0.2 * (1 - rel_plot_width), 1))

    fig.savefig(Path(__file__).parent / f"{Path(__file__).name[:-3]}.pdf")


EXPERIMENTS = {
    **branin_ablation_num_meta_tasks.EXPERIMENTS,
    **branin_ablation_num_points_per_task.EXPERIMENTS,
    **hartmann6_ablation_num_meta_tasks.EXPERIMENTS,
    **hartmann6_ablation_num_points_per_task.EXPERIMENTS,
}

if __name__ == "__main__":
    main(EXPERIMENTS)
