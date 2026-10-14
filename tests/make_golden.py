"""Regenerate the golden regression files: python tests/make_golden.py [case ...]"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regression_cases  # noqa: E402

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden')

if __name__ == '__main__':
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    names = sys.argv[1:] or list(regression_cases.CASES)
    for name in names:
        arrays = regression_cases.CASES[name]()
        np.savez(os.path.join(GOLDEN_DIR, name + '.npz'), **arrays)
        print('%-28s %s' % (name, ', '.join('%s%s' % (k, v.shape) for k, v in arrays.items() if k == 'populations' or k.startswith('wait'))))
