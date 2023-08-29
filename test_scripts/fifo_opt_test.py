from pathlib import Path
import pickle

from src.analysis.latency.engine_v2 import build_dfg

from lightningsim.model import Solution
from lightningsim.runner import Runner
from asyncio import run

from src.opt_fifo import solve_fifo_optimization
from src.opt_random import solve_random_opt
from src.opt_init import get_depths_for_minimum_latency_64


SOLUTION_FP = Path(
    "/usr/scratch/skaram7/inr_dsp_builds/my_model_64/model_top_vitis_hls_project/solution1/"
)

if __name__ == "__main__":
    dfg_pkl_fp = Path("./dfg_graph/dfg_graph.pkl")

    if not dfg_pkl_fp.exists():
        print("Cached DFG pkl file not found, building DFG from scratch")

        solution = Solution(SOLUTION_FP)
        runner = Runner(solution)
        trace = run(runner.run())

        dfg = build_dfg(trace)
        dfg_pkl_fp.parent.mkdir(parents=True, exist_ok=True)
        with open(dfg_pkl_fp, "wb") as f:
            pickle.dump(dfg, f)
    else:
        print("Loading DFG from cached pkl file")
        with open(dfg_pkl_fp, "rb") as f:
            dfg = pickle.load(f)

    init_dict = get_depths_for_minimum_latency_64(dfg)
    solve_fifo_optimization(dfg, n_jobs=48, init_dict=init_dict)
    # solve_random_opt(dfg, n_jobs=48, init_dict=init_dict)
