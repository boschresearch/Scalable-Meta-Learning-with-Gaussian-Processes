import numpy as np

from scamlgp.benchmarking.benchmarks import Quadratic
from scamlgp.benchmarking.utils import (
    add_noise_to_meta_data_objectives,
    get_benchmarks,
    get_benchmarks_with_output_dimensions,
    get_benchmarks_with_search_space_dimensions,
)


def test_benchmarks_found():
    benchmarks = get_benchmarks()
    assert len(benchmarks) > 0


def test_1d_output_benchmarks_found():
    benchmarks = get_benchmarks_with_output_dimensions(dimensions=1)
    assert len(benchmarks) > 0


def test_1d_search_space_benchmarks_found():
    benchmarks = get_benchmarks_with_search_space_dimensions(dimensions=1)
    assert len(benchmarks) > 0


def test_add_noise_to_meta_data_objectives_absolute_noise():
    seed = 42
    noise_scale = 0.1
    benchmark = Quadratic(n_data_per_task=[8] * 8)
    meta_data = benchmark.get_meta_data(distribution="random")

    noisy_meta_data = add_noise_to_meta_data_objectives(
        meta_data, noise_scale=noise_scale, seed=seed
    )
    ref_rng = np.random.default_rng(seed)

    for task_id in noisy_meta_data.keys():
        for raw, noisy in zip(meta_data[task_id], noisy_meta_data[task_id]):
            np.testing.assert_almost_equal(
                noisy.objectives["loss"],
                raw.objectives["loss"] + noise_scale * ref_rng.standard_normal(),
            )
