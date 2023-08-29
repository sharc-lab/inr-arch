from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.pntx import (
    PointCrossover,
    SinglePointCrossover,
    TwoPointCrossover,
)
from pymoo.operators.mutation.pm import PM
from pymoo.operators.mutation.gauss import GaussianMutation
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling


import numpy as np
import tqdm

from rich.pretty import pprint as pp


from src.analysis.latency.engine_v2 import DFG


from joblib import Parallel, delayed


def fifo_depth_objective(x: np.ndarray):
    depths_sum = np.sum(x, axis=1)
    return depths_sum


def latency_objective(x: np.ndarray, dfg: DFG):
    total_latency = np.max(dfg.with_depths(
        {stream: depth for stream, depth in zip(dfg.streams, x)}
    ).get_latency())
    return total_latency


def deadlock_constraint(x: np.ndarray, dfg: DFG):
    c, diff = dfg.with_depths(
        {stream: depth for stream, depth in zip(dfg.streams, x)}
    ).has_cycle_approx()
    return int(diff)


class FIFOProblem(Problem):
    def __init__(self, dfg: DFG, n_jobs: int = 1, **kwargs):
        self.dfg = dfg
        self.min_fifo_depth = 2
        self.max_fifo_depth = 66000

        self.n_jobs = n_jobs

        super().__init__(
            n_var=len(self.dfg.streams),
            n_obj=2,
            n_constr=1,
            xl=self.min_fifo_depth,
            xu=self.max_fifo_depth,
            type_var=int,
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        x_split = [x[i] for i in range(x.shape[0])]

        f = fifo_depth_objective(x)

        f_latency_split = Parallel(n_jobs=self.n_jobs, backend="loky", verbose=0)(
            delayed(latency_objective)(x, self.dfg) for x in x_split
        )
        f_latency = np.array(f_latency_split)

        out["F"] = [f, f_latency]
        # print(f)

        g_deadlock_split = Parallel(n_jobs=self.n_jobs, backend="loky", verbose=0)(
            delayed(deadlock_constraint)(x, self.dfg) for x in x_split
        )
        g_deadlock = np.array(g_deadlock_split)

        out["G"] = [g_deadlock]

        return out


def solve_fifo_optimization(dfg: DFG, n_jobs: int = 1, init_dict={}):
    x0 = (np.ones((48, len(dfg.streams))) * 500).astype(int)
    for stream, depth in init_dict.items():
        x0[:, dfg.streams.index(stream)] = depth

    problem = FIFOProblem(dfg, n_jobs=n_jobs)
    algorithm = NSGA2(
        pop_size=50,
        eliminate_duplicates=True,
        # crossover=SBX(prob=0.8, eta=0.01, vtype=float, repair=RoundingRepair()),
        crossover=PointCrossover(
            prob=1.0, n_points=4, vtype=float, repair=RoundingRepair()
        ),
        # mutation=PM(prob=1.0, eta=0.01, vtype=float, repair=RoundingRepair()),
        mutation=GaussianMutation(
            sigma=1000, prob=1.0, vtype=float, repair=RoundingRepair()
        ),
        sampling=x0,
    )
    res = minimize(problem, algorithm, termination=("n_gen", 10), seed=1, verbose=True)
    pp(res.__dict__)

    return res
