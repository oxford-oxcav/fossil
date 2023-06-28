import timeit

import torch

from src.cegis import Cegis
import src.plotting as plotting
from experiments import analysis
from src import consts
from src.cegis_supervisor import CegisSupervisorQ

"""Top-level module for running benchmarks, and (eventually) using a CLI"""


N_PROCS = 4
BASE_SEED = 167


def run_benchmark(
    cegis_options: consts.CegisConfig,
    repeat=1,
    record=False,
    plot=False,
    concurrent=False,
):
    """Unified interface for running benchmarks with a fixed seed.

    Allows for running benchmarks with different configurations regarding recording, plotting, and concurrency.

    Args:
        cegis_options (CegisConfig): Cegis configuration
        repeat (int, optional): How many times to repeat over consecutive seeds. Defaults to 1.
        record (bool, optional): record to csv. Defaults to False.
        plot (bool, optional): plot benchmark. Defaults to False.
        concurrent (bool, optional): For each attempt, run multiple seeds in parallel and take first successful result. Defaults to False.
    """
    torch.set_num_threads(1)

    for i in range(repeat):
        torch.manual_seed(BASE_SEED + i * N_PROCS)
        start = timeit.default_timer()
        if concurrent:
            c = CegisSupervisorQ(max_P=N_PROCS)
            result = c.solve(cegis_options)
        else:
            c = Cegis(cegis_options)
            result = c.solve()
        stop = timeit.default_timer()
        T = stop - start
        print("Elapsed Time: {}".format(T))

        if plot:
            if cegis_options.N_VARS != 2:
                raise NotImplementedError(
                    "Plotting is only supported for 2-dimensional problems"
                )
            plotting.benchmark(result.f, result.cert, domains=cegis_options.DOMAINS)
        if record:
            rec = analysis.Recorder()
            rec.record(cegis_options, result, T)
