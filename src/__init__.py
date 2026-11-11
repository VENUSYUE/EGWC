"""Public API for edge group walk centrality and greedy edge selection.

Numerical imports are deferred so command-line tools can configure native
thread limits before loading NumPy and Numba.
"""
from importlib import import_module

__version__ = "0.1.0"
__all__ = [
    "Graph", "load", "approx_singleton", "exact_singleton_scores",
    "egwc", "gains_from_entries", "exact_entries", "exact_gains",
    "exact_greedy", "schur_round", "schur_greedy", "pcg",
]


def __getattr__(name):
    if name in __all__:
        value = getattr(import_module(".core", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(set(globals()) | set(__all__))
