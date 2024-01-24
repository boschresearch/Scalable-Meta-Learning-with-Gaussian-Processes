import itertools
from functools import partial

import pytest
from blackboxopt import EvaluationSpecification

from scamlgp.benchmarking.benchmarks.base import _shgo_minimize
from scamlgp.benchmarking.utils import (
    get_benchmarks,
    get_benchmarks_with_output_dimensions,
    get_benchmarks_with_search_space_dimensions,
)

from .utils import assert_dict_equals, assert_metadata_equal

BENCHMARK_CLASSES = get_benchmarks()
BENCHMARK_CLASSES_1D_INPUT = get_benchmarks_with_search_space_dimensions(1)
BENCHMARK_CLASSES_1D_OUTPUT = get_benchmarks_with_output_dimensions(1)

META_DATA_DISTRIBUTIONS = ["random", "sobol"]


@pytest.mark.parametrize("benchmark_class", BENCHMARK_CLASSES)
def test_tasks_with_fixed_seed(benchmark_class):
    seed = 3
    num_points = 8
    num_tasks = 4
    n_data_per_task = [num_points] * num_tasks

    b1 = benchmark_class(n_data_per_task, seed=seed)
    b2 = benchmark_class(n_data_per_task, seed=seed)

    # Target tasks are different
    with pytest.raises(AssertionError):
        assert_dict_equals(b1.target_task.__dict__, b2.target_task.__dict__)

    # Meta tasks are the same
    for id, task1 in b1.meta_tasks.items():
        task2 = b2.meta_tasks[id]
        assert_dict_equals(task1.__dict__, task2.__dict__)


@pytest.mark.parametrize("benchmark_class", BENCHMARK_CLASSES_1D_OUTPUT)
def test_tasks_optimum_with_shgo(benchmark_class):
    seed = 3
    num_points = 8
    num_tasks = 4
    n_data_per_task = [num_points] * num_tasks
    b = benchmark_class(n_data_per_task, seed=seed)

    fun = partial(
        b._numpy_wrapper_call,
        context=b.target_task.context,
        settings=b.target_task.settings,
    )

    result = _shgo_minimize(fun, b.search_space)
    assert b.optimum == pytest.approx(result.fun, rel=1e-3)


@pytest.mark.parametrize(
    "benchmark_class, distribution",
    itertools.product(BENCHMARK_CLASSES, META_DATA_DISTRIBUTIONS),
)
def test_the_same_metadata_with_fixed_seed(benchmark_class, distribution):
    seed = 3
    num_points = 8
    num_tasks = 2

    bm = benchmark_class(n_data_per_task=[num_points] * num_tasks)

    if len(bm.search_space) > 1 and distribution == "optimized_1d":
        with pytest.raises(AssertionError):
            bm.get_meta_data(seed=seed, distribution=distribution)

    else:
        metadata1 = bm.get_meta_data(seed=seed, distribution=distribution)
        metadata2 = bm.get_meta_data(seed=seed, distribution=distribution)
        assert_metadata_equal(metadata1, metadata2)


@pytest.mark.parametrize(
    "benchmark_class, distribution",
    itertools.product(BENCHMARK_CLASSES, META_DATA_DISTRIBUTIONS),
)
def test_different_metadata_with_not_fixed_seed(benchmark_class, distribution):
    num_points = 8
    num_tasks = 2

    bm = benchmark_class(n_data_per_task=[num_points] * num_tasks)

    if len(bm.search_space) > 1 and distribution == "optimized_1d":
        with pytest.raises(AssertionError):
            bm.get_meta_data(distribution=distribution)
    else:
        metadata1 = bm.get_meta_data(distribution=distribution)
        metadata2 = bm.get_meta_data(distribution=distribution)

        if distribution not in ["optimized_1d", "optimized"]:
            with pytest.raises(AssertionError):
                assert_metadata_equal(metadata1, metadata2)


@pytest.mark.parametrize("benchmark_class", BENCHMARK_CLASSES)
def test_output_dimensions_match_metadata_dimensions(benchmark_class):
    num_points = 8
    num_tasks = 2

    bm = benchmark_class(n_data_per_task=[num_points] * num_tasks)
    bm.get_meta_data(distribution="random")

    assert bm.output_dimensions == 1


@pytest.mark.parametrize("benchmark_class", BENCHMARK_CLASSES)
def test_numpy_wrapper_call(benchmark_class):
    num_points = 32
    for _ in range(num_points):
        bm = benchmark_class()

        configuration = bm.search_space.sample()
        numerical_conf = bm.search_space.to_numerical(configuration)

        eval_spec = EvaluationSpecification(configuration=configuration)
        res = bm(eval_spec)

        assert (
            bm._numpy_wrapper_call(
                numerical_conf,
                context=bm.target_task.context,
                settings=bm.target_task.settings,
            )
            == res.objectives["loss"]
        )
