# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import copy
import warnings
from typing import Dict, Hashable, Iterable, Optional, Union

import parameterspace as ps
import torch
from blackboxopt import Evaluation, Objective, sort_evaluations
from blackboxopt.logger import logger
from blackboxopt.optimizers.botorch_base import impute_nans_with_constant, to_numerical
from botorch import fit
from botorch.acquisition import UpperConfidenceBound as _UpperConfidenceBound
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions import ModelFittingError
from botorch.exceptions.warnings import BotorchWarning
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.utils.datasets import SupervisedDataset
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.utils.warnings import GPInputWarning


# This function is adapted from BoTorch
# https://github.com/pytorch/botorch/blob/7ce7c6d9d36c0eeefd9e15bdd0355d41e16f575d/botorch/optim/utils.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# The original source code is licensed under the MIT license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.
def sample_all_priors(model: GPyTorchModel, num_retries: int = 5) -> None:
    r"""Sample from hyperparameter priors (in-place).
    Args:
        model: A GPyTorchModel.
        num_retries: Number of sampling attempts.
    """

    for prior_name, module, prior, closure, setting_closure in model.named_priors():
        if setting_closure is None:
            raise RuntimeError(
                "Must provide inverse transform to be able to sample from prior."
            )

        parameter_name = "_".join(prior_name.split(".")[-1].split("_")[:-1])
        constraint = module.constraint_for_parameter_name("raw_" + parameter_name)

        for i in range(num_retries + 1):
            try:
                sample = prior.sample(closure(module).shape)
            except NotImplementedError:
                warnings.warn(
                    f"`rsample` not implemented for {type(prior)}. Skipping.",
                    BotorchWarning,
                )
                break

            is_valid = constraint.inverse_transform(sample).isfinite().all()

            if is_valid:
                setting_closure(module, sample)
                break

        if i == num_retries:
            raise RuntimeError(
                f"Sampling of {prior_name} failed {num_retries} times. Please check "
                f"the compatibility between prior support and the constraint."
            )
        elif i > 0:
            logger.warning(f"Sampling from {prior_name} failed {i} times in a row.")


def metadata_to_numerical(
    meta_data: Dict[Hashable, Iterable[Evaluation]],
    search_space: ps.ParameterSpace,
    objective: Objective,
    batch_shape: torch.Size = torch.Size(),
    torch_dtype: torch.dtype = torch.float32,
) -> Dict[Hashable, SupervisedDataset]:
    """Convert meta evaluations to torch tensors.

    Evaluations for each task are sorted before the conversion, to ensure that
    optimization runs are deterministic regardless of the order of input meta
    evaluations.

    Args:
        meta_data: Dictionary of meta evaluations, where each task ID of `Hashable`
            type contains a list of evaluations.
        search_space: Search space used during optimization.
        objective: Objective that was used for optimization.
        batch_shape: Batch dimension(s) used for batched models.
        torch_dtype: Type of returned tensors.

    Returns:
        Meta-data represented as torch tensors.
    """
    metadata_numerical = {}
    for task_id, task_data in meta_data.items():
        X_raw, Y = to_numerical(
            sort_evaluations(task_data),
            search_space,
            [objective],
            batch_shape=batch_shape,
            torch_dtype=torch_dtype,
        )
        # fill in NaNs originating from inactive parameters (conditional spaces support)
        X = impute_nans_with_constant(X_raw)

        metadata_numerical[task_id] = SupervisedDataset(X, Y)
    return metadata_numerical


def validate_meta_data(meta_data: Dict[Hashable, SupervisedDataset]):
    """Validate the metadata passed to meta-learning algorithms"""
    if len(meta_data) == 0:
        raise ValueError("Empty meta data. Needs at least one source task.")
    task_id_source_0, data_source_0 = list(meta_data.items())[0]
    X_shape = data_source_0.X.shape
    Y_shape = data_source_0.Y.shape
    if X_shape[:-2] != Y_shape[:-2]:
        raise ValueError(
            f"The X and Y batch sizes of task {task_id_source_0} are not equal."
        )
    for task_id, task_data in meta_data.items():
        if (
            task_data.X.shape[:-2] != X_shape[:-2]
            or task_data.Y.shape[:-2] != Y_shape[:-2]
            or task_data.X.shape[-1] != X_shape[-1]
        ):
            raise ValueError(
                f"Dimensions of tasks {task_id_source_0} and {task_id} do not match."
            )
        if task_data.Y.shape[-1] != 1:
            raise ValueError(
                f"The output dimension of task {task_id} is {task_data.Y.shape[-1]} "
                f"but must be one"
            )


def optimize_marginal_likelihood(
    model: Union[GPyTorchModel, ExactGP],
    num_restarts: int = 0,
    **fit_gpytorch_options,
):
    """Refit underlying model's hyperparameters by maximizing the marginal log
    likelihood on the training data.

    The hyperparameters are optimized `num_restarts + 1` times. The first time,
    the optimization is warm-started with the current parameters. Otherwise, the
    initial conditions are sampled from the priors.

    Raises:
        ValueError: If all attempts at optimizing hyperparameters failed.
    """
    # Ensure the underlying botorch mll fitting doesn't retry and while at it resample
    # the priors, because that's done here
    fit_gpytorch_options["max_attempts"] = 1

    # Ensure that all errors bubble up, and the underlying botorch mll fitting doesn't
    # catch any of them
    fit_gpytorch_options["caught_exception_types"] = ()

    # When `model.__call__` method is called, GPyTorch tries to warn us that we
    # perform inference at the training points.
    # The warning is not relevant here, so we can safely filter it.
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=GPInputWarning)

        mll_val = -float("inf")
        model_state_dict = copy.deepcopy(model.state_dict())

        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # The first restart is warm start
        try:
            fit.fit_gpytorch_mll(mll, **fit_gpytorch_options)
            with torch.no_grad():
                mll_val = mll(model(*model.train_inputs), model.train_targets)
            # store copy of state dict, as it contains only "pointers" to the values!
            model_state_dict = copy.deepcopy(model.state_dict())
        except (RuntimeError, ModelFittingError) as e:
            logger.warning(
                f"Error occurred while optimizing model's hyperparameters: `{e}` "
                "Will continue with restarts to find a stable set of hyperparameters."
            )

        # For the remaining restarts, we sample the hyperparameters from priors
        for _ in range(num_restarts):
            try:
                sample_all_priors(model)
                fit.fit_gpytorch_mll(mll, **fit_gpytorch_options)
                with torch.no_grad():
                    mll_temp = mll(model(*model.train_inputs), model.train_targets)
            except (RuntimeError, ModelFittingError) as e:
                logger.warning(
                    f"Error occurred while optimizing the model hyperparameters: `{e}` "
                    "This restart will be skipped."
                )
                continue

            if mll_temp > mll_val:
                mll_val = mll_temp
                # COPY state dict again for reference values!
                model_state_dict = copy.deepcopy(model.state_dict())

        model.load_state_dict(model_state_dict)

        if mll_val == -float("inf"):
            raise ModelFittingError(
                "Hyperparameter optimization failed for all attempts. Usually this "
                "indicates a problem with model's input data or hyperparameter priors "
                "definitions."
            )


class UpperConfidenceBound(_UpperConfidenceBound):
    def __init__(
        self,
        model: Model,
        beta: Union[float, torch.Tensor] = 9.0,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs,
    ) -> None:
        """UCB with default `beta=9.0` and always set to `maximize=False`."""
        super().__init__(model, beta, posterior_transform, maximize=False, **kwargs)
