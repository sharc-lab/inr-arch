import numpy as np

from src.analysis.latency.engine_v2 import DFG




def deadlock_constraint(x: np.ndarray, dfg: DFG):
    c, diff = dfg.with_depths(
        {stream: depth for stream, depth in zip(dfg.streams, x)}
    ).has_cycle_approx()
    return int(diff)


def solve_random_opt(dfg: DFG, n_jobs=48, init_dict={}):
    DEFAULT_DEPTH = 2
    x_init = np.ones((len(dfg.streams),)) * MIN_DEPTH
    for stream, depth in init_dict.items():
        x_init[dfg.streams.index(stream)] = depth

    print(x_init)
