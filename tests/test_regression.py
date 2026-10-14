"""Bit-for-bit regression of every library code path against the stored golden runs."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regression_cases  # noqa: E402

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden')


@pytest.mark.parametrize('name', sorted(regression_cases.CASES))
def test_regression(name):
    golden = np.load(os.path.join(GOLDEN_DIR, name + '.npz'))
    fresh = regression_cases.CASES[name]()
    assert set(fresh) == set(golden.files)
    for key in golden.files:
        np.testing.assert_array_equal(fresh[key], golden[key], err_msg='%s: %s' % (name, key))
