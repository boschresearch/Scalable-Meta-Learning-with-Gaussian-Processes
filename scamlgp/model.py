# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import copy
from itertools import compress
from typing import Dict, Hashable, List, Optional, Tuple, Type, Union

import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.utils.datasets import SupervisedDataset
from gpytorch import Module
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from gpytorch.means import ZeroMean
from gpytorch.priors import GammaPrior, LogNormalPrior
from linear_operator import to_linear_operator
from linear_operator.operators import PsdSumLinearOperator

from scamlgp.utils import optimize_marginal_likelihood, validate_meta_data


def _get_default_likelihood(
    batch_shape: torch.Size = torch.Size(),
) -> GaussianLikelihood:
    noise_prior = LogNormalPrior(-8.0, 2.0)
    return GaussianLikelihood(
        noise_prior=noise_prior,
        noise_constraint=Interval(1e-8, 1e-2, initial_value=1e-3),
        batch_shape=batch_shape,
    )


def _get_kernel_source_gp(
    base_kernel: Union[Type[RBFKernel], Type[MaternKernel]],
    ard_num_dims: int = 1,
    batch_shape: torch.Size = torch.Size(),
) -> ScaleKernel:
    lengthscale_prior = GammaPrior(3.0, 6.0)  # Same prior as `SingleTaskGP`
    outputscale_prior = GammaPrior(2.0, 0.15)  # Same prior as `SingleTaskGP`

    return ScaleKernel(
        base_kernel(
            ard_num_dims=ard_num_dims,
            lengthscale_prior=lengthscale_prior,
            # If the outputscale is zero, lengthscales tend to be arbitrary. This
            # avoids numerical instability when lengthscales get very small/large.
            # Since our domain is in [0, 1] the lengthscale ranges allow for
            # anything between
            lengthscale_constraint=Interval(
                lower_bound=torch.tensor(1e-4),
                upper_bound=torch.tensor(1e2),
                initial_value=torch.full((ard_num_dims,), 0.5),
            ),
            batch_shape=batch_shape,
        ),
        outputscale_prior=outputscale_prior,
        # Avoid numerical instability: prior.log_prob is nan for output_scale = 0
        # If the scale gets too large then we have to add more noise to make the
        # matrix PSD. Since our model acts on normalized data the maximum output
        # scale should be ~1.
        outputscale_constraint=Interval(
            lower_bound=torch.tensor(1e-4),
            upper_bound=torch.tensor(1e2),
            initial_value=torch.tensor(1.0),
        ),
        batch_shape=batch_shape,
    )


def _get_default_kernel(
    base_kernel: Union[Type[RBFKernel], Type[MaternKernel]],
    ard_num_dims: int = 1,
    batch_shape: torch.Size = torch.Size(),
) -> ScaleKernel:
    # The lengthscale prior is not as restrictive as for `SingleTaskGP`
    # since the difference signal can have a broad range of lengthscales depending on
    # how successful the transfer is. This prior is inspired by our study on
    # meta-learning priors.
    lengthscale_prior = LogNormalPrior(0.5, 1.5)
    # The outputscale parameter can attain a broad range of values because the target
    # y-values are normalized wrt all target and meta observations. This prior is
    # inspired by our study on meta-learning priors.
    outputscale_prior = LogNormalPrior(-2.0, 3.0)
    return ScaleKernel(
        base_kernel(
            ard_num_dims=ard_num_dims,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=Interval(
                lower_bound=torch.tensor(1e-4),
                upper_bound=torch.tensor(1e2),
                initial_value=torch.full((ard_num_dims,), 1.0),
            ),
            batch_shape=batch_shape,
        ),
        outputscale_prior=outputscale_prior,
        outputscale_constraint=Interval(
            lower_bound=torch.tensor(1e-4),
            upper_bound=torch.tensor(1e2),
            initial_value=torch.tensor(0.1),
        ),
        batch_shape=batch_shape,
    )


def _compute_target_prior(
    x: torch.Tensor, source_gps: List[SingleTaskGP], weights: torch.Tensor
) -> Tuple[torch.Tensor, PsdSumLinearOperator]:
    """Compute the prior distribution of the target GP.

    Args:
        x: A `n x d` or `batch_shape x n x d` (batch mode) tensor of features.
        source_gps: Dictionary containing the source GPs (order needs to match weights).
        weights: The source weights (order needs to match source_gps).

    Returns:
        Tuple containing:
            - The mean function as `n x 1` or `batch_shape x n x 1` tensor.
            - The covariance function as `n x n` or `batch_shape x n x n` lazy tensor.
    """
    if len(source_gps) != len(weights):
        raise ValueError(
            f"The number of source GPs, {len(source_gps)}, does not "
            f"equal the number of weights, {len(weights)}"
        )
    posteriors = [task_gp.posterior(x) for task_gp in source_gps]
    weighted_post_mean_source = [p.mvn.mean * w for p, w in zip(posteriors, weights)]
    weighted_post_cov_source = [
        p.mvn.lazy_covariance_matrix * w**2 for p, w in zip(posteriors, weights)
    ]
    prior_mean_target = sum(weighted_post_mean_source, torch.zeros(*x.shape[:-1]))
    prior_cov_target = PsdSumLinearOperator(*weighted_post_cov_source)
    return prior_mean_target.unsqueeze(-1), prior_cov_target


def meta_fit_scamlgp(
    meta_data: Dict[Hashable, SupervisedDataset],
    likelihood: Optional[Likelihood] = None,
    covar_module: Optional[Module] = None,
    num_restarts_log_likelihood: int = 5,
    seed: Optional[int] = None,
) -> Dict[Hashable, SingleTaskGP]:
    """Train the source GPs on the given meta-data.

    Args:
        meta_data: Dictionary containing the source datasets.
        likelihood: A likelihood. If omitted, use a standard
            `GaussianLikelihood` with inferred noise level.
        covar_module: The kernel of the GPs in the stack. If omitted, use an
            `RBFKernel`.
        num_restarts_log_likelihood: The number of restarts for the log-likelihood
            optimization of the model parameters. The first restart corresponds to
            warm-starting, while the rest of the initial conditions are sampled
            from the parameters' prior.
        seed: A seed to make the optimization reproducible.

    Returns:
        A dictionary of the trained source GPs and corresponding task IDs.

    """
    if seed is not None:
        torch.manual_seed(seed=seed)
    validate_meta_data(meta_data)
    source_gp_0 = list(meta_data.values())[0]
    ard_num_dims = source_gp_0.X.shape[-1]
    batch_shape = source_gp_0.X.shape[:-2]
    if likelihood is None:
        likelihood = _get_default_likelihood(batch_shape=batch_shape)
    if covar_module is None:
        covar_module = _get_kernel_source_gp(
            base_kernel=RBFKernel, ard_num_dims=ard_num_dims, batch_shape=batch_shape
        )
    source_gps = {}
    for task_id, task_data in meta_data.items():
        likelihood = copy.deepcopy(likelihood)
        covar_module = copy.deepcopy(covar_module)
        task_gp = SingleTaskGP(
            train_X=task_data.X(),
            train_Y=task_data.Y(),
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=ZeroMean(batch_shape=batch_shape),
            outcome_transform=Standardize(1, batch_shape=batch_shape),
        )
        optimize_marginal_likelihood(task_gp, num_restarts=num_restarts_log_likelihood)
        source_gps[task_id] = task_gp
    return source_gps


def significant_weights_mask(
    weights: torch.Tensor, std_Y_vals: torch.Tensor, threshold: float
) -> torch.Tensor:
    r"""Boolean mask of weights that when rescaled exceed the threshold.

    Args:
        weights: The weights of the models. shape = (n_source_tasks,)
        std_Y_vals: The standard deviation of the models' Y-values.
            shape = (n_source_tasks,)
        threshold: threshold for weight pruning, $\tau$, according to the criterion
            $$
                \frac{w_i \sigma_i}{\sum_j w_j \sigma_j}n_w < \tau,
            $$
            where $w_i$ is the weight of task $i$, $\sigma_i$ is the standard
            deviation of the Y-values from task $i$, and $n_w$ denotes the number of
            weights.

    Returns:
        Mask containing True for significant weights and False otherwise.
    """
    num_weights = len(weights)
    w_times_sigma = weights * std_Y_vals
    norm_weights = w_times_sigma * num_weights / w_times_sigma.sum()
    return norm_weights >= threshold


class ScaMLGP(SingleTaskGP):
    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        source_gps: Dict[Hashable, SingleTaskGP],
        likelihood: Optional[Likelihood] = None,
        covar_module: Optional[Module] = None,
        weight_pruning_threshold: float = 1e-3,
    ) -> None:
        r"""Scalable and Modular Kernel for Transfer Learning with Gaussian Processes.

        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            source_gps: Dictionary containing the source GPs.
            likelihood: A likelihood. If omitted, use a standard
                `GaussianLikelihood` with inferred noise level.
            covar_module: The kernel of the target GP. If omitted, use an `RBFKernel`.
            weight_pruning_threshold: Relative threshold for weight pruning during
                prediction. Weights satisfying the following condition are pruned:
                `weights * num_weights/ weights.sum() < weight_pruning_threshold`.
                Defaults to 1e-3. Note that the absolute threshold decreases linearly
                with `num_weights`. Usually, it is not required to change this.

        Example:
            >>> pi = torch.pi
            >>> meta_X = torch.distributions.Uniform(-pi, pi).sample((20, 1))
            >>> meta_Y1 = torch.sin(meta_X)
            >>> meta_Y2 = torch.cos(meta_X)
            >>> meta_data = {
            >>>     0: SupervisedDataset(meta_X, meta_Y1),
            >>>     1: SupervisedDataset(meta_X, meta_Y2),
            >>>}
            >>> source_models = meta_fit_scamlgp(meta_data)
            >>> X = torch.distributions.Uniform(-torch.pi, torch.pi).sample((7, 1))
            >>> Y = torch.sin(X) * X/10
            >>> model = ScaMLGP(X, Y, source_models)
        """
        self._weight_pruning_threshold = weight_pruning_threshold

        batch_shape = train_Y.shape[:-2]
        n_source_tasks = len(source_gps)
        # Note that, since we normalize with respect to all meta- and target data, the
        # target data alone does generally not have mean = 0 and std = 1. This may lead
        # to BoTorch raising warnings about Y-normalization and can be safely ignored.
        Y_meta = torch.cat(
            [
                gp.outcome_transform.untransform(gp.train_targets.unsqueeze(-1))[0]
                for gp in list(source_gps.values())
            ],
            dim=-2,
        )
        Y_all = torch.cat([Y_meta, train_Y], dim=-2)
        outcome_transform = Standardize(1, batch_shape=batch_shape)
        outcome_transform(Y_all)
        # Make sure parent constructor doesn't re-learn the normalizer and only consumes
        # current state
        outcome_transform.eval()

        # Cache the posterior mean of the source models
        if train_Y.shape[-2] > 0:
            with torch.no_grad():
                posteriors = [gp.posterior(train_X) for gp in source_gps.values()]
                self.source_means = torch.stack(
                    [p.mvn.mean for p in posteriors], dim=-1
                )
                # No need to deal with LazyTensor since the target typically has little
                # data
                self.source_covs = torch.stack(
                    [p.mvn.covariance_matrix for p in posteriors], dim=-1
                )

        if covar_module is None:
            covar_module = _get_default_kernel(
                base_kernel=RBFKernel,
                ard_num_dims=train_X.shape[-1],
                batch_shape=batch_shape,
            )

        self.source_gps = source_gps
        batch_shape = train_Y.shape[:-2]
        if likelihood is None:
            likelihood = _get_default_likelihood(batch_shape=batch_shape)
        # We normalize to zero-mean unit-variance
        mean_module = ZeroMean(batch_shape=batch_shape)
        if outcome_transform is None:
            outcome_transform = Standardize(1, batch_shape=batch_shape)
        # If input is empty, do not standardize the data
        if train_Y.nelement() == 0:
            outcome_transform = None
        super().__init__(
            train_X,
            train_Y,
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=outcome_transform,
        )

        # Add weight parameters
        self.register_parameter(
            "raw_weights",
            torch.nn.Parameter(torch.full((n_source_tasks,), 1.0 / n_source_tasks)),
        )
        # Use a generic distribution that is constant for `w \lesssim 1` and decays
        # quickly for `w \gtrsim 1`. This prior is best for tasks functions that live on
        # relatively similar scales.
        self.register_prior(
            "weights_prior",
            GammaPrior(1.0, 1.0),
            lambda m: m.weights,
            lambda m, v: m._set_weights(v),  # pylint: disable=protected-access
        )
        # For unrelated tasks, the weights can reach a value of zero and may conflict
        # with a strictly positive prior. We therefore bound the weights from below.
        weights_constraint = GreaterThan(1e-10, transform=None)
        self.register_constraint(
            "raw_weights",
            weights_constraint,
        )
        self.to(train_X)

    @property
    def weights(self):
        # When accessing the parameter, apply the constraint transform
        return self.raw_weights_constraint.transform(self.raw_weights)

    @weights.setter
    def weights(self, value):
        self._set_weights(value)

    def _set_weights(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_weights)
        # When setting the parameter, transform the actual value to a raw one by
        # applying the inverse transform
        self.initialize(
            raw_weights=self.raw_weights_constraint.inverse_transform(value)
        )

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        if self.training:
            # Use cached values
            mean = (self.source_means @ self.weights).unsqueeze(-1)
            cov = self.source_covs @ self.weights**2
        else:
            std_Y_vals = torch.tensor(
                [m.outcome_transform.stdvs.flatten() for m in self.source_gps.values()]
            )
            mask = significant_weights_mask(
                self.weights, std_Y_vals, self._weight_pruning_threshold
            )
            significant_weights = self.weights[mask]
            significant_models = list(compress(self.source_gps.values(), mask))
            mean, cov = _compute_target_prior(
                x=x, source_gps=significant_models, weights=significant_weights
            )
        if hasattr(self, "outcome_transform"):
            # Transform the source terms to the target space
            self.outcome_transform.eval()
            mean = self.outcome_transform(mean)[0]
            cov /= self.outcome_transform.stdvs**2
            if self.training:  # return to training mode, if relevant
                self.outcome_transform.train()
        cov = to_linear_operator(cov) + self.covar_module(x)
        return MultivariateNormal(mean.squeeze(-1), cov)
