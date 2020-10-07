import traceback
from functools import partial

import torch
import timeit

from experiments.benchmarks.benchmarks_bc import elementary
from src.barrier.cegis_barrier import Cegis
from src.shared.activations import ActivationType
from src.shared.consts import VerifierType, LearnerType
from src.shared.cegis_values import CegisConfig
from src.plots.plot_lyap import plot_lyce
import numpy as np


def main():

    batch_size = 500
    system = partial(elementary, batch_size)
    activations = [ActivationType.LIN_SQUARE]
    hidden_neurons = [10]
    opts = {CegisConfig.SP_SIMPLIFY.k: True, CegisConfig.SP_HANDLE.k: False}
    try:
        start = timeit.default_timer()
        c = Cegis(2, LearnerType.NN, VerifierType.DREAL, activations, system, hidden_neurons,
                  **opts)
        state, vars, f, iters = c.solve()
        end = timeit.default_timer()

        print('Elapsed Time: {}'.format(end - start))
        print("Found? {}".format(state['found']))
    except Exception as _:
        print(traceback.format_exc())


if __name__ == '__main__':
    torch.manual_seed(167)
    main()
