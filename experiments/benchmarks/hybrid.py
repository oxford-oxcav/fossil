from experiments.benchmarks.benchmarks_bc import twod_hybrid
from src.shared.consts import VerifierType, LearnerType
from src.shared.activations import ActivationType
from src.shared.cegis_values import CegisConfig
from src.barrier.cegis_barrier import Cegis
from src.plots.plot_lyap import plot_lyce

from functools import partial
import numpy as np
import traceback
import timeit
import torch


def main():

    batch_size = 500
    system = partial(twod_hybrid, batch_size)
    activations = [ActivationType.LIN_SQUARE]
    hidden_neurons = [3] * len(activations)
    opts = {CegisConfig.SP_SIMPLIFY.k: False, CegisConfig.SP_HANDLE.k: False,
            CegisConfig.SYMMETRIC_BELT.k: False}
    try:
        start = timeit.default_timer()
        c = Cegis(2, LearnerType.NN, VerifierType.Z3, activations, system, hidden_neurons,
                  **opts)
        state, vars, f_learner, iters = c.solve()
        end = timeit.default_timer()

        print('Elapsed Time: {}'.format(end - start))
        print("Found? {}".format(state['found']))
    except Exception as _:
        print(traceback.format_exc())


if __name__ == '__main__':
    torch.manual_seed(167)
    main()