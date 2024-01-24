import parameterspace as ps
import pytest
from blackboxopt import Objective
from blackboxopt.optimizers.testing import (
    ALL_REFERENCE_TESTS,
    handles_conditional_space,
    is_deterministic_when_reporting_shuffled_evaluations,
    optimize_single_parameter_sequentially_for_n_max_evaluations,
    respects_fixed_parameter,
)

from scamlgp.optimizer import ScaMLGPBO
from scamlgp.testing import META_OPTIMIZER_REFERENCE_TESTS
from scamlgp.utils import UpperConfidenceBound

from .meta_data_examples import (
    META_DATA_1D_SPACE,
    META_DATA_CONDITIONAL_SPACE,
    META_DATA_FIXED_PARAM_SPACE,
    META_DATA_MIXED_SPACE,
    generate_forrester_meta_data,
)


@pytest.mark.parametrize(
    "reference_test", ALL_REFERENCE_TESTS + META_OPTIMIZER_REFERENCE_TESTS
)
def test_all_reference_tests(reference_test, seed):
    reference_test_kwargs = {"seed": seed}
    if reference_test is optimize_single_parameter_sequentially_for_n_max_evaluations:
        # Set this lower than the default of 20 to speed the test up
        reference_test_kwargs["n_max_evaluations"] = 6

    if reference_test is respects_fixed_parameter:
        meta_data = META_DATA_FIXED_PARAM_SPACE
    elif reference_test is handles_conditional_space:
        # Set this lower than the default of 10 to speed the test up
        reference_test_kwargs["n_max_evaluations"] = 2
        meta_data = META_DATA_CONDITIONAL_SPACE
    elif reference_test is is_deterministic_when_reporting_shuffled_evaluations:
        meta_data = META_DATA_1D_SPACE
    else:
        meta_data = META_DATA_MIXED_SPACE

    reference_test(
        ScaMLGPBO,
        dict(
            meta_data=meta_data,
            num_initial_random_samples=1,
            max_pending_evaluations=5,
        ),
        **reference_test_kwargs
    )


def test_report_evaluation_with_missing_objective():
    """Evaluations with missing objective are ignored during model fit.
    However, these evaluations are kept within the optimizer."""
    space = ps.ParameterSpace()
    space.add(ps.ContinuousParameter("x", (0, 1)))
    objective = Objective("loss", greater_is_better=False)

    source_descriptors = [{"a": 0.95, "b": 0.02, "c": 1}]
    metadata = generate_forrester_meta_data(
        search_space=space,
        objective=objective,
        num_source_points=32,
        source_descriptors=source_descriptors,
    )

    opt = ScaMLGPBO(
        search_space=space,
        objective=objective,
        meta_data=metadata,
        acquisition_function_factory=UpperConfidenceBound,
        num_initial_random_samples=1,
        num_restarts_log_likelihood=2,
        max_pending_evaluations=5,
    )

    n_evaluations = 5
    evaluations = []
    for i in range(n_evaluations):
        eval_spec = opt.generate_evaluation_specification()
        evaluation = eval_spec.create_evaluation(objectives={"loss": 0.42 + 0.1 * i})
        evaluations.append(evaluation)

    evaluations[-2].objectives["loss"] = None
    opt.report(evaluations)

    opt.generate_evaluation_specification()
    assert len(opt.pending_specifications) == 1

    assert opt.X.numel() == n_evaluations
    assert opt.losses.numel() == n_evaluations
    assert opt.model.train_inputs[0].numel() == n_evaluations - 1
    assert opt.model.train_targets.numel() == n_evaluations - 1
