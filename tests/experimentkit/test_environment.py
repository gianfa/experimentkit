# %%
"""

python -m pytest tests/experimentkit/test_environment.py -vv --pdb -s
"""
import sys
sys.path += ["../.."] # good to test in jupyter

import matplotlib.pyplot as plt

from experimentkit.environment import get_system_info


def test_get_system_info():
    sinfo = get_system_info()

    assert type(sinfo) == dict