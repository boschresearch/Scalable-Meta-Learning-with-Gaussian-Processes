import numpy as np
import pytest

from scamlgp.benchmarking.benchmarks import Branin, Hartmann3D, Hartmann6D
from scamlgp.benchmarking.benchmarks.api import Task

HARTMANN3D = {
    "fun": Hartmann3D,
    "meta": {
        "x_min": {"x1": 0.114614, "x2": 0.555649, "x3": 0.852547},
        "f_min": -3.86278,
        "descriptors": {"alpha1": 1.0, "alpha2": 1.2, "alpha3": 3.0, "alpha4": 3.2},
    },
}

HARTMANN6D = {
    "fun": Hartmann6D,
    "meta": {
        "x_min": {
            "x1": 0.20169,
            "x2": 0.150011,
            "x3": 0.476874,
            "x4": 0.275332,
            "x5": 0.311652,
            "x6": 0.6573,
        },
        "f_min": -3.32237,
        "descriptors": {"alpha1": 1.0, "alpha2": 1.2, "alpha3": 3.0, "alpha4": 3.2},
    },
}

BRANIN2D = {
    "fun": Branin,
    "meta": {
        "x_min": {
            "x1": -np.pi,
            "x2": 12.275,
        },
        "f_min": 0.397887,
        "descriptors": {
            "a": 1.0,
            "b": 5.1 / (4 * np.pi**2),
            "c": 5 / np.pi,
        },
        "settings": {
            "r": 6,
            "s": 10,
        },
        "context": {"t": 1 / (8 * np.pi)},
    },
}

BENCHMARKS = (HARTMANN3D, HARTMANN6D, BRANIN2D)


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_known_and_actual_minima(benchmark):
    """Does known minimum coincide with the minimum returned by the functions?"""
    seed = 3
    n_data_per_task = []

    settings = benchmark["meta"].get("settings", {})
    context = benchmark["meta"].get("context", {})

    params_fun = {
        **benchmark["meta"]["x_min"],
        **benchmark["meta"]["descriptors"],
        **settings,
        **context,
    }
    b = benchmark["fun"](n_data_per_task=n_data_per_task, seed=seed)

    known_minimum = benchmark["meta"]["f_min"]
    actual_minimum = b.function(**params_fun)

    np.testing.assert_almost_equal(known_minimum, actual_minimum, decimal=4)

    # finally compare to the minimum that the benchmark class provides using
    # scipy.optimize
    for k, v in benchmark["meta"]["descriptors"].items():
        b.target_task.descriptors[k] = v

        for k, v in benchmark["meta"].get("settings", {}).items():
            b.target_task.settings[k] = v

        for k, v in benchmark["meta"].get("context", {}).items():
            b.target_task.context[k] = v

    np.testing.assert_almost_equal(b.optimum, actual_minimum, decimal=4)


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_known_and_calculated_minima(benchmark):
    """Does the known minimum coincide with the minimum found by a global optimizer?"""
    seed = 3
    n_data_per_task = []

    b = benchmark["fun"](n_data_per_task=n_data_per_task, seed=seed)

    # fix target task
    b._target_task = Task(
        uid="test_instance",
        descriptors=benchmark["meta"]["descriptors"],
        settings=benchmark["meta"].get("settings", {}),
        context=benchmark["meta"].get("context", {}),
    )

    known_minimum = benchmark["meta"]["f_min"]
    calculated_minimum = b.optimum

    np.testing.assert_almost_equal(known_minimum, calculated_minimum, decimal=4)
