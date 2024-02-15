""" Environment and profiling functions
"""

import platform
import re
import uuid, json, psutil, logging
import sys
import timeit
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt


def getsizeof_in(var, targetUnit):
    """Get size of a variable in specific units of measurement."""
    from sys import getsizeof

    shifts_to = {
        "Kb": 7,
        "Mb": 17,
        "Gb": 27,
        "KB": 10,
        "MB": 20,
        "GB": 30,
        "TB": 40,
        "PB": 50,
    }
    return getsizeof(var) / float(1 << shifts_to[targetUnit])


def get_system_info(mac_address: bool = False) -> dict:
    """Get System info"""
    info = {}
    info["platform"] = platform.system()
    info["platform_release"] = platform.release()
    info["platform_version"] = platform.version()
    info["architecture"] = platform.machine()

    if mac_address:
        info["mac_address"] = ":".join(re.findall("..", "%012x" % uuid.getnode()))

    info["processor"] = platform.processor()
    info["cpu_freq"] = str(psutil.cpu_freq())
    info["cpu_count"] = str(psutil.cpu_count())
    info["python_implementation"] = str(platform.python_implementation())
    info["python_compiler"] = str(platform.python_compiler())

    info["ram"] = str(round(psutil.virtual_memory().total / (1024.0**3))) + " GB"
    return info


def get_precision_info() -> dict:
    info = {}
    info["int_max_size"] = sys.maxsize
    # TODO: break down float_info
    info["float_info"] = np.finfo(float)
    info["int_info"] = np.iinfo(int)
    return info


def measureit(f: callable, iterations: Iterable[int] = [1, 10, 50, 100]) -> np.ndarray:
    return {n_i: timeit.timeit(f, number=100) for n_i in iterations}


def plot_measures_dict(
    d: dict, xlabel: str = "# iterations", ylabel: str = "t[s]", ax: plt.Axes = None
):
    """Plot measurements"""
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(list(d.keys()), list(d.values()))
    ax.set(title="Execution time", xlabel=xlabel, ylabel=ylabel)
    return ax
