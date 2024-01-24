"""Helper functions for testing."""

from typing import Dict, List, Union

import numpy as np
from blackboxopt import Evaluation


def assert_dict_equals(d1, d2):
    """
    Raises an AssertionError if two dictionaries are not equal in one side
    comparison.

    One-side comparison means that for every `d1` key `d2` should have the same key and
    equal value. Note that in this way `d2` could have some values that are not in `d1`,
    but assert will pass. In case both side asserts needed, the function should be
    called twice by user, with a switched order on the second call of the input
    dictionaries.

    The methods extends pytest dictionaries assert, by injecting valid asserts if more
    complex inner structures, such as ndarray and list.

    Parameters
    ----------
    d1 : dict
        First dictionary.
    d2 : dict
        Second dictionary.

    Raises
    ------
    AssertionError
        An exception is thrown if dictionaries differ.
    """

    assert set(d1.keys()) == set(d2.keys())

    for k, d1k in d1.items():
        if isinstance(d1k, np.ndarray):
            np.testing.assert_array_almost_equal(d1k, d2[k])
        elif isinstance(d1k, list):
            if isinstance(d1k[0], dict):
                for dd1k, dd2k in zip(d1k, d2[k]):
                    assert_dict_equals(dd1k, dd2k)
            else:
                np.testing.assert_array_almost_equal(d1k, d2[k])
        elif isinstance(d1k, dict):
            assert_dict_equals(d1k, d2[k])
        else:
            assert d1k == d2[k], "Key: {}, v1: {}, v2: {}".format(k, d1k, d2[k])


def assert_metadata_equal(
    md1: Dict[Union[str, int], List[Evaluation]],
    md2: Dict[Union[str, int], List[Evaluation]],
):
    """Corresponding method to compare return values of get_metadata

    Parameters
    ----------
    md1 : First meta-data set.
    md2 : Second meta-data set.

    Raises
    ------
    AssertionError
        An exception is thrown if dictionaries differ.
    """

    assert set(md1.keys()) == set(md2.keys())
    for task_id in md1.keys():
        assert len(md1[task_id]) == len(md2[task_id])
        for d1, d2 in zip(md1[task_id], md2[task_id]):
            for attribute in ["objectives", "settings", "context", "configuration"]:
                assert getattr(d1, attribute) == getattr(d2, attribute)
