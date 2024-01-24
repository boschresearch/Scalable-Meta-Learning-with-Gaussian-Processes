import parameterspace as ps

from scamlgp.benchmarking.benchmarks.base import Base

from .utils import assert_dict_equals


def test_create_the_same_task_with_fixed_seed():
    seed = 3
    task_id = "test_task"

    descriptors = ps.ParameterSpace()
    descriptors.add(ps.ContinuousParameter(name="a", bounds=[0.5, 1.5]))
    descriptors.add(ps.ContinuousParameter(name="b", bounds=[-0.9, 0.9]))
    descriptors.add(ps.IntegerParameter(name="c", bounds=[-1, 1]))

    settings = ps.ParameterSpace()
    settings.add(ps.ContinuousParameter(name="d", bounds=[-2.0, 1.0]))
    settings.add(ps.ContinuousParameter(name="e", bounds=[3.0, 10.0]))

    context = ps.ParameterSpace()
    context.add(ps.ContinuousParameter(name="f", bounds=[6.0, 7.0]))
    context.add(
        ps.OrdinalParameter(name="g", values=["freezing cold", "cold", "warm", "hot"])
    )
    context.add(
        ps.CategoricalParameter(name="h", values=["some text", "new info", 42, 89.5])
    )

    t1 = Base.create_random_task(task_id, descriptors, settings, context, seed=seed)
    t2 = Base.create_random_task(task_id, descriptors, settings, context, seed=seed)

    assert_dict_equals(t1.__dict__, t2.__dict__)
