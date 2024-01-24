# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import json
import warnings
from typing import Dict, Iterable, List, Optional, Tuple, Union
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from blackboxopt import Evaluation, Objective
from scipy.stats import sem

from scamlgp.benchmarking.experiment_config_utils import (
    Experiment,
    parse_experiment_config,
)


def compute_regrets(
    objective: Objective, optimum: float, objective_values: List[dict]
) -> List[float]:
    """Compute regrets for every step of the optimization problem.

    Args:
        objective: The objective to use for regret computation.
        optimum: The known optimum with respect to which to compute the regret.
        objective_values: Dictionaries containing the objective values at each
            iteration, each including a key for the given `objective.name` and the
            respective objective value.
    """
    sign = -1.0 if objective.greater_is_better else 1.0

    regrets: List[float] = []
    for ovs in objective_values:
        loss = sign * ovs[objective.name]

        regret = loss - (sign * optimum)
        # For some benchmarks the minimum is determined using another optimizer.
        # Therefore, in this case, the minimum used here is generally not exactly the
        # true minimum. Hence, (very small) negative regrets are possible.
        if regret < -1e-6:
            warnings.warn(
                f"A negative regret was detected. The regret value was {regret}.",
                Warning,
            )
        if not regrets:
            regrets.append(regret)
        else:
            regrets.append(min(regret, regrets[-1]))

    return regrets


def _regrets_from_studies(studies):
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
    return regrets


def _plot_regrets(
    ax: plt.Axes,
    studies: list,
    color_primary,
    color_secondary,
    robust_statistics: bool,
    linestyle: str = "-",
    label: Optional[str] = None,
    optimum: Optional[Union[float, List[float]]] = None,
):
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
            optimum=(
                s["optimum"]
                if optimum is None
                else (optimum[i] if isinstance(optimum, Iterable) else optimum)
            ),
            objective_values=[Evaluation(**e).objectives for e in s["evaluations"]],
        )
        for i, s in enumerate(studies)
    ]

    if not regrets:
        warn(f"No regrets for {label}")
        return

    n_trials = len(regrets[0])
    x = np.arange(1, n_trials + 1)

    if robust_statistics:
        center_statistic = np.median(regrets, axis=0).squeeze()
        upper_bound = np.quantile(regrets, 0.75, axis=0)
        lower_bound = np.quantile(regrets, 0.25, axis=0)
    else:
        center_statistic = np.mean(regrets, axis=0).squeeze()
        regrets_sem = sem(regrets, axis=0).squeeze()
        upper_bound = center_statistic + regrets_sem
        lower_bound = center_statistic - regrets_sem

    ax.set_yscale("log")
    ax.plot(
        x,
        center_statistic,
        color=color_primary,
        linestyle=linestyle,
        # label=f"{label} (S{len(studies)})",
        label=f"{label}",
    )
    ax.fill_between(x=x, y1=lower_bound, y2=upper_bound, color=color_secondary)
    ax.set_xlim(1, n_trials)


def _plot_objective(
    ax: plt.Axes,
    studies: list,
    color_primary,
    color_secondary,
    objective: Objective,
    robust_statistics: bool,
    linestyle: str = "-",
    label: Optional[str] = None,
):
    objective_values = np.array(
        [
            [Evaluation(**e).objectives[objective.name] for e in s["evaluations"]]
            for s in studies
        ]
    )

    if not len(objective_values):
        warn(f"No objective values for {label}")
        return

    n_trials = len(objective_values[0])
    x = np.arange(1, n_trials + 1)

    objective_values = (
        np.maximum.accumulate(objective_values, axis=1)
        if objective.greater_is_better
        else np.minimum.accumulate(objective_values, axis=1)
    )

    if robust_statistics:
        center_statistic = np.median(objective_values, axis=0).squeeze()
        upper_bound = np.quantile(objective_values, 0.75, axis=0)
        lower_bound = np.quantile(objective_values, 0.25, axis=0)
    else:
        center_statistic = np.mean(objective_values, axis=0).squeeze()
        regrets_sem = sem(objective_values, axis=0).squeeze()
        upper_bound = center_statistic + regrets_sem
        lower_bound = center_statistic - regrets_sem

    ax.plot(
        x,
        center_statistic,
        color=color_primary,
        linestyle=linestyle,
        label=f"{label} (S{len(studies)})",
    )
    ax.fill_between(x=x, y1=lower_bound, y2=upper_bound, color=color_secondary)
    ax.set_xlim(1, n_trials)


def _study_wise_optima(data: List[dict], objective: Objective) -> List[float]:
    optima = []
    # Number of studies that are present for everyone, because some runs might have
    # failed studies and thus a lower number of study results
    max_n_studies = max(len(d["studies"]) for d in data)
    min_or_max = max if objective.greater_is_better else min
    for i_study in range(max_n_studies):
        optima.append(
            min_or_max(
                min_or_max(
                    e["objectives"][objective.name]
                    for e in d["studies"][i_study]["evaluations"]
                )
                for d in data
                if i_study < len(d["studies"])
            )
        )
    return optima


def grouped_results(
    runs_data: list,
    optimizer_styles: dict,
    groups: Dict[str, Iterable[Experiment]],
    robust_statistics: bool,
    objective: Union[Objective, List[Objective]],
    optimum: Optional[float] = None,
    use_regrets: bool = True,
    use_benchmark_optimum: bool = True,
    rel_fig_width: float = 1.0,
    fig_height: float = 4.0,
    x_limits: Optional[List[Tuple[int, int]]] = None,
    y_limits: Optional[List[Tuple[float, float]]] = None,
    n_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
    sharey: str = "none",
    h_pad: float = 1.8,
) -> plt.Figure:
    plt.rc("font", family="serif")
    if n_rows is None:
        n_rows = 2
    if n_cols is None:
        n_cols = int(np.ceil(len(groups) / n_rows))
    fig, axs = plt.subplots(
        min(n_rows, len(groups)),
        n_cols,
        figsize=(6.75 * rel_fig_width, fig_height),
        sharex="col",
        sharey=sharey,
    )
    if not isinstance(axs, Iterable):
        axs = np.array([axs])

    run_configs = [Experiment(**run["experiment_config"]) for run in runs_data]
    for i, ((title, group), ax) in enumerate(zip(groups.items(), axs.flatten())):
        ax.set_title(title)

        _objective = objective[i] if isinstance(objective, list) else objective
        study_wise_optima = (
            _study_wise_optima(
                [runs_data[run_configs.index(config)] for config in group], _objective
            )
            if use_regrets and not use_benchmark_optimum
            else []
        )

        # group contains a tuple of configs
        for config in group:
            try:
                i_config = run_configs.index(config)
            except ValueError:
                print(
                    "Unable to find configuration in available results, skipping",
                    json.dumps(parse_experiment_config(config.__dict__), indent=2),
                )
                continue
            data = runs_data[i_config]

            optimizer_style = (
                optimizer_styles[config.optimizer["cls"]]
                if isinstance(config.optimizer, dict)
                else optimizer_styles[config.optimizer]
            )

            if use_regrets:
                _plot_regrets(
                    ax=ax,
                    studies=data["studies"],
                    robust_statistics=robust_statistics,
                    color_primary=(*optimizer_style["color"], 0.8),
                    color_secondary=(*optimizer_style["color"], 0.3),
                    linestyle=optimizer_style["line"],
                    label=optimizer_style["label"],
                    optimum=(
                        study_wise_optima
                        if optimum is None and not use_benchmark_optimum
                        else optimum
                    ),
                )
            else:
                _plot_objective(
                    ax=ax,
                    studies=data["studies"],
                    robust_statistics=robust_statistics,
                    color_primary=(*optimizer_style["color"], 0.8),
                    color_secondary=(*optimizer_style["color"], 0.3),
                    linestyle=optimizer_style["line"],
                    label=optimizer_style["label"],
                    objective=_objective,
                )
        # Start ticking at 1 instead of at 0; also 0 is hidden due to xlim [1, n_trials]
        ax.set_xticks([1] + list(ax.get_xticks()[1:]))

    y_label = "Regret" if use_regrets else _objective.name

    if len(axs.shape) == 2:
        for ax in axs[:, 0]:
            ax.set_ylabel(y_label)
        for ax in axs[-1, :]:
            ax.set_xlabel("Iteration")
    elif len(axs.shape) == 1:
        axs[0].set_ylabel(y_label)
        for ax in axs:
            ax.set_xlabel("Iteration")
    else:
        raise ValueError(f"Incompatible axis shape {axs.shape}")

    handles, labels = [], []
    for ax in axs.flatten():
        handle, label = ax.get_legend_handles_labels()
        handles.extend(handle)
        labels.extend(label)
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="center right",
        ncol=1,
        labelspacing=1.5,
        frameon=False,
    )
    for i, ax in enumerate(axs.flatten()):
        if x_limits is not None:
            ax.set_xlim(*x_limits[i])
        if y_limits is not None:
            ax.set_ylim(*y_limits[i])
    fig.tight_layout(h_pad=h_pad, rect=(0.0, 0.0, 0.8 - 0.2 * (1 - rel_fig_width), 1))

    return fig


def _compute_average_cum_regret(studies: List[dict]) -> float:
    regrets = _regrets_from_studies(studies)
    cum_regret = np.sum(regrets, axis=1).squeeze()
    return np.mean(cum_regret).item()


def _compute_sem_cum_regret(studies: List[dict]) -> float:
    regrets = _regrets_from_studies(studies)
    cum_regret = np.sum(regrets, axis=1).squeeze()
    return np.sqrt(np.var(cum_regret) / cum_regret.shape[0]).item()


def plot_meta_data_summary_comparison(
    results: dict,
    styles: dict,
    ax: plt.Axes,
    num_meta_tasks: Optional[int] = None,
    num_points_per_task: Optional[int] = None,
):
    assert (num_meta_tasks is None and num_points_per_task is not None) or (
        num_meta_tasks is not None and num_points_per_task is None
    )
    plt.rc("font", family="serif")

    _r = {k: v for k, v in results.items() if k != "environment"}
    df = pd.json_normalize(_r.values())

    df = df.assign(
        num_meta_tasks=df["experiment_config.benchmark.kwargs.n_data_per_task"].map(len)
    )
    df = df.assign(
        num_points_per_task=df[
            "experiment_config.benchmark.kwargs.n_data_per_task"
        ].map(lambda x: x[0] if len(x) > 0 else 0)
    )

    # Fill in missing class attribute if just an optimizer class and no full kwargs are
    # given
    _mask = df["experiment_config.optimizer.cls"].isna()
    df.loc[_mask, "experiment_config.optimizer.cls"] = df[
        "experiment_config.optimizer"
    ][_mask]

    if num_meta_tasks is not None:
        df = df[(df["num_meta_tasks"] == num_meta_tasks) | (df["num_meta_tasks"] == 0)]

    if num_points_per_task is not None:
        df = df[
            (df["num_points_per_task"] == num_points_per_task)
            | (df["num_points_per_task"] == 0)
        ]

    mode = "num_points_per_task" if num_meta_tasks is not None else "num_meta_tasks"

    # iterating over x values in the figure
    plot_data = []
    for x_value, group in df.groupby(mode):
        # each row in group is one optimizer
        plot_data.extend(
            [
                {
                    mode: x_value,
                    "average_cum_regret": average_cum_regret,
                    "sem_cum_regret": sem_cum_regret,
                    "experiment_config.optimizer.cls": optimizer_cls,
                }
                for average_cum_regret, sem_cum_regret, optimizer_cls in zip(
                    group["studies"].map(_compute_average_cum_regret),
                    group["studies"].map(_compute_sem_cum_regret),
                    group["experiment_config.optimizer.cls"],
                )
            ]
        )
    # plot meta-learning baselines
    plot_df = pd.DataFrame(plot_data)
    plot_styles = {f"{k.__module__}.{k.__name__}": v for k, v in styles.items()}
    for optimizer_cls, group in plot_df.groupby("experiment_config.optimizer.cls"):
        ax.errorbar(
            group[mode],
            group["average_cum_regret"],
            yerr=group["sem_cum_regret"],
            capsize=2,
            ls=plot_styles[optimizer_cls]["line"],
            color=plot_styles[optimizer_cls]["color"],
            label=plot_styles[optimizer_cls]["label"],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
