# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import logging
from typing import Any, Callable, Dict, Hashable, Iterable, Optional, Union

import parameterspace as ps
import torch
from blackboxopt import Evaluation, Objective
from blackboxopt.optimizers.botorch_base import (
    SingleObjectiveBOTorchOptimizer,
    filter_y_nans,
)
from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model
from gpytorch import Module
from gpytorch.likelihoods import Likelihood

from scamlgp.model import ScaMLGP, meta_fit_scamlgp
from scamlgp.utils import (
    UpperConfidenceBound,
    metadata_to_numerical,
    optimize_marginal_likelihood,
)


class ScaMLGPBO(SingleObjectiveBOTorchOptimizer):
    def __init__(
        self,
        search_space: ps.ParameterSpace,
        objective: Objective,
        meta_data: Dict[Hashable, Iterable[Evaluation]],
        gp_likelihood: Optional[Likelihood] = None,
        gp_kernel: Optional[Module] = None,
        base_gp_kernel: Optional[Module] = None,
        acquisition_function_factory: Optional[
            Callable[[Model], AcquisitionFunction]
        ] = None,
        af_optimizer_kwargs: Optional[dict] = None,
        num_initial_random_samples: int = 0,
        max_pending_evaluations: Optional[int] = 1,
        num_restarts_log_likelihood: int = 5,
        model_kwargs: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
        seed: Optional[int] = None,
        torch_dtype: torch.dtype = torch.float64,
    ):
        r"""Single objective meta-learning BO optimizer that uses ScaMLGP as surrogate
        model.

        ScaMLGP stands for Scalable and Modular Kernel for Transfer Learning with
        Gaussian Processes. The model is a GP that models the source and target data
        jointly with one single kernel. ScaMLGP has a flexible Ansatz and should
        yield excellent results on most transfer-learning problems. The technique is
        scalable in the number of source tasks.

        The technique works by fitting individual GPs to each source data set. Their
        posterior predictions are then weighted linearly and given to the prior of
        the target GP.

        The joint-kernel representation of ScaMLGP is relatively complicated and not
        shown here. The prior probability distribution of the target conditioned on the
        source data, $\mathcal{D}_s$, is given by

        $$
            p[f_t(x) | \mathcal{D}_s] = \mathcal{N}[\mu_s(x), \Sigma_s(x,x) + k_t(x,x)],
        $$

        where $\mu_s(x)$ is prior mean function given by

        $$
            \mu_s(x) = \sum_{i=1}^{n_s}w_i\mu_i(x),
        $$

        where $n_s$ denotes the total number of source tasks, $w_i$ the weight of the
        $i$-th source, and  $\mu_i(x)$ is the mean of the posterior probability
        distribution of the $i$-th source GP. $\Sigma_s(x,x)$ is the weighted
        posterior covariance of the source models

        $$
            \Sigma_s(x) = \sum_{i=1}^{n_s}w_i^2\Sigma_i(x,x).
        $$

        The weights are optimized jointly with the marginal log-likelihood of the target
        data.

        Args:
            search_space: The space in which to optimize.
            objective: The objective to optimize.
            meta_data: Meta-data to meta train the model on.
            gp_likelihood: The likelihood function. Defaults to a Gaussian Likelihood.
            gp_kernel: The kernel of the GPs in the stack. Defaults to an RBF kernel.
            acquisition_function_factory: Callable that produces an acquisition function
                instance, could also be a compatible acquisition function class. If not
                provided, Upper Confidence Bound with 9.0 exploration margin is used.
                Only acquisition functions to be minimized are supported.
                Providing a partially initialized class is possible with, e.g.
                `functools.partial(ExpectedImprovement, exploration_margin=0.05)`.
            af_optimizer_kwargs: Settings for acquisition function optimizer,
                see `botorch.optim.optimize_acqf`.
            num_initial_random_samples: Size of the initial space-filling design that
                is used before starting BO. The points are sampled randomly in the
                search space. If no random sampling is required, set it to 0. Some
                acquisition functions, like EI and PI, don't work with this
                parameter being zero, since they need some data to work with.
            max_pending_evaluations: Maximum number of parallel evaluations. For
                sequential BO use the default value of 1. If no limit is required,
                set it to None.
            num_restarts_log_likelihood: The number of restarts for the log-likelihood
                optimization of the model parameters. The first restart corresponds to
                warm-starting, while the rest of the initial conditions are sampled
                from the parameters' prior.
            model_kwargs: Model-specific arguments. See model documentation for details.
            logger: Custom logger.
            seed: A seed to make the optimization reproducible.
            torch_dtype: Torch data type used for storing the data. This needs to match
                the dtype of the model used. `float64` is recommended for GP-based
                methods.
        """
        n_features = len(search_space)
        batch_shape = torch.Size()
        if acquisition_function_factory is None:
            acquisition_function_factory = UpperConfidenceBound
        metadata_numerical = metadata_to_numerical(
            meta_data, search_space, objective, batch_shape, torch_dtype
        )
        self.num_restarts_log_likelihood = num_restarts_log_likelihood
        self.source_gps = meta_fit_scamlgp(
            metadata_numerical,
            likelihood=gp_likelihood,
            covar_module=base_gp_kernel,
            seed=seed,
        )
        self.model_kwargs = {} if model_kwargs is None else model_kwargs
        model = ScaMLGP(
            train_X=torch.empty((*batch_shape, 0, n_features), dtype=torch_dtype),
            train_Y=torch.empty((*batch_shape, 0, 1), dtype=torch_dtype),
            source_gps=self.source_gps,
            likelihood=gp_likelihood,
            covar_module=gp_kernel,
        )
        super().__init__(
            search_space=search_space,
            objective=objective,
            model=model,
            acquisition_function_factory=acquisition_function_factory,
            af_optimizer_kwargs=af_optimizer_kwargs,
            num_initial_random_samples=num_initial_random_samples,
            max_pending_evaluations=max_pending_evaluations,
            batch_shape=batch_shape,
            logger=logger,
            seed=seed,
            torch_dtype=torch_dtype,
        )

    def report(self, evaluations: Union[Evaluation, Iterable[Evaluation]]):
        """Validate evaluations, do the bookkeeping and potentially refit underlying
        model's hyperparameters on reported evaluations.

        Please refer to the docstring of
        `blackboxopt.base.SingleObjectiveOptimizer.report` for a description of the
        method.
        """
        _evals = evaluations if isinstance(evaluations, list) else [evaluations]
        super()._update_internal_evaluation_data(_evals)

        if len(self.X) < self.num_initial_random:
            return

        # filter data with unknown objective
        x_filtered, y_filtered = filter_y_nans(self.X, self.losses)
        # skip model fit if all targets were unknown
        if x_filtered.numel() == 0:
            return

        self.model = ScaMLGP(
            x_filtered,
            y_filtered,
            self.source_gps,
            likelihood=self.model.likelihood,
            covar_module=self.model.covar_module,
            **self.model_kwargs,
        )

        optimize_marginal_likelihood(self.model, self.num_restarts_log_likelihood)
