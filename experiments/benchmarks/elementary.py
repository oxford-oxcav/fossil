# pylint: disable=not-callable
import traceback
from functools import partial

import torch
import timeit

from experiments.benchmarks.benchmarks_bc import elementary
from src.barrier.cegis_barrier import Cegis
from src.shared.activations import ActivationType
from src.shared.cegis_values import CegisConfig
from src.shared.consts import VerifierType, LearnerType, TrajectoriserType, RegulariserType
from src.shared.cegis_values import CegisConfig
from src.plots.plot_lyap import plot_lyce
import numpy as np


def main():
    batch_size = 500
    system = partial(elementary, batch_size)
    activations = [ActivationType.LIN_SQUARE]
    hidden_neurons = [10]
    opts = {
        CegisConfig.N_VARS.k: 2,
        CegisConfig.LEARNER.k: LearnerType.NN,
        CegisConfig.VERIFIER.k: VerifierType.DREAL,
        CegisConfig.TRAJECTORISER.k: TrajectoriserType.DEFAULT,
        CegisConfig.REGULARISER.k: RegulariserType.DEFAULT,
        CegisConfig.ACTIVATION.k: activations,
        CegisConfig.SYSTEM.k: system,
        CegisConfig.N_HIDDEN_NEURONS.k: hidden_neurons,
        CegisConfig.SP_SIMPLIFY.k: True,
        CegisConfig.SP_HANDLE.k: False,
        CegisConfig.SYMMETRIC_BELT.k: False,
    }

    start = timeit.default_timer()
    c = Cegis(**opts)
    state, vars, f, iters = c.solve()
    end = timeit.default_timer()

    print('Elapsed Time: {}'.format(end - start))
    print("Found? {}".format(state['found']))

    # plotting -- only for 2-d systems
    if len(vars) == 2 and state['found']:
        plot_lyce(np.array(vars), state['V'],
                  state['V_dot'], f)


if __name__ == '__main__':
    torch.manual_seed(167)
    main()
