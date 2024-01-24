import numpy as np
import pytest
from blackboxopt import EvaluationSpecification

from scamlgp.benchmarking.benchmarks import Quadratic
from scamlgp.benchmarking.noise.benchmark import NoisyBenchmark
from scamlgp.benchmarking.noise.homoscedastic import HomoscedasticGaussianNoise


def test_noisy_benchmark_with_homoscedastic_gaussian_noise(n_samples=128):
    """Tests the call to the NoisyBenchmark is equivalent to applying the Noise to the
    noise-free Evaluation.
    """
    seed = 42
    benchmark = Quadratic(seed=seed)
    noise = HomoscedasticGaussianNoise(noise_std={"loss": 1.0}, seed=seed)
    noisy_benchmark = NoisyBenchmark(benchmark, noise)

    rng1 = np.random.default_rng(seed)

    for i in range(n_samples):
        config = benchmark.search_space.sample()
        eval_spec = EvaluationSpecification(configuration=config)

        noise_free_eval = benchmark(eval_spec)
        noisy_eval = noisy_benchmark(eval_spec)

        alternative_noisy_eval = noise(noise_free_eval, rng=rng1)
        assert alternative_noisy_eval.objectives == noisy_eval.objectives


def test_seeding_noisy_benchmark_with_homoscedastic_gaussian_noise(n_samples=128):
    """Tests that two instances with the same seed yield same results"""
    seed = 42
    benchmark = Quadratic(seed=seed)
    noise1 = HomoscedasticGaussianNoise(noise_std={"loss": 1.0}, seed=seed)
    noise2 = HomoscedasticGaussianNoise(noise_std={"loss": 1.0}, seed=seed)
    noisy_benchmark1 = NoisyBenchmark(benchmark, noise1)
    noisy_benchmark2 = NoisyBenchmark(benchmark, noise2)

    for i in range(n_samples):
        config = benchmark.search_space.sample()
        eval_spec = EvaluationSpecification(configuration=config)
        eval1 = noisy_benchmark1(eval_spec)
        eval2 = noisy_benchmark2(eval_spec)

        assert eval1.objectives == eval2.objectives


def test_zero_noise_for_homoscedastic_gaussian_noise():
    """Test that setting the noise to zero is equivalent to the noise free benchmark."""
    seed = 42
    benchmark = Quadratic(seed=seed)
    no_noise = HomoscedasticGaussianNoise(noise_std={"loss": 0.0}, seed=seed)
    noise_free_benchmark = NoisyBenchmark(benchmark, no_noise)

    noise_free_meta_data1 = benchmark.get_meta_data(distribution="random", seed=seed)
    noise_free_meta_data2 = noise_free_benchmark.get_meta_data(
        distribution="random", seed=seed
    )

    for task_id in noise_free_meta_data1.keys():
        for e1, e2 in zip(
            noise_free_meta_data1[task_id],
            noise_free_meta_data2[task_id],
        ):
            assert e1.configuration == e2.configuration
            assert e1.objectives == e2.objectives


def test_get_meta_data_noisy_benchmark_with_homoscedastic_gaussian_noise():
    """Test that the meta-data configurations with a fixed seed are identical, while the
    objectives are different assuming the benchmark was initialized with the same seed.
    """
    seed = 42
    benchmark = Quadratic(seed=seed)

    noise = HomoscedasticGaussianNoise(noise_std={"loss": 1.0}, seed=seed)
    noisy_benchmark = NoisyBenchmark(benchmark, noise)

    clean_meta_data = benchmark.get_meta_data(distribution="random", seed=seed)
    noisy_meta_data = noisy_benchmark.get_meta_data(distribution="random", seed=seed)

    for task_id in clean_meta_data.keys():
        for e1, e2 in zip(
            noisy_meta_data[task_id],
            clean_meta_data[task_id],
        ):
            assert e1.configuration == e2.configuration
            assert e1.objectives != e2.objectives


def test_missing_noise_keys():
    seed = 42
    benchmark = Quadratic(seed=seed)

    invalid_noise = HomoscedasticGaussianNoise(noise_std={"not_loss": 1.0}, seed=seed)
    noisy_benchmark = NoisyBenchmark(benchmark, invalid_noise)

    with pytest.raises(KeyError):
        noisy_benchmark.get_meta_data(distribution="random")


def test_unused_noise_keys():
    seed = 42
    benchmark = Quadratic(seed=seed)

    invalid_noise = HomoscedasticGaussianNoise(
        noise_std={"loss": 0.5, "not_loss": 1.0}, seed=seed
    )
    noisy_benchmark = NoisyBenchmark(benchmark, invalid_noise)
    noisy_benchmark.get_meta_data(distribution="random")


def test_get_optimum_independent_of_noise():
    seed = 42
    benchmark = Quadratic(seed=seed)

    noise = HomoscedasticGaussianNoise(noise_std={"loss": 0.1}, seed=seed)
    noisy_benchmark = NoisyBenchmark(benchmark, noise)
    min1 = benchmark.optimum
    min2 = noisy_benchmark.optimum

    assert min1 == pytest.approx(min2, 1e-4)


def test_noisy_benchmark_properties():
    benchmark = Quadratic(seed=42)
    noise = HomoscedasticGaussianNoise(noise_std={"loss": 0.1})
    noisy_benchmark = NoisyBenchmark(benchmark, noise)

    assert benchmark.target_task == noisy_benchmark.target_task
    assert benchmark.meta_tasks == noisy_benchmark.meta_tasks
    assert benchmark.output_dimensions == noisy_benchmark.output_dimensions
    assert benchmark.objectives == noisy_benchmark.objectives


def test_noise_repr():
    noise = HomoscedasticGaussianNoise(noise_std={"loss": 0.1}, seed=42)
    assert "HomoscedasticGaussianNoise(noise_std={'loss': 0.1}, seed=42)" == str(noise)

    noise = HomoscedasticGaussianNoise(noise_std={"loss": 0.1}, seed=None)
    assert "HomoscedasticGaussianNoise(noise_std={'loss': 0.1}, seed=None)" == str(
        noise
    )
