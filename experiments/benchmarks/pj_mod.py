import traceback
from functools import partial

import torch
import numpy as np
import timeit

from experiments.benchmarks.benchmarks_bc import prajna07_modified
from src.barrier.cegis_barrier import Cegis
from src.shared.activations import ActivationType
from src.shared.cegis_values import CegisConfig
from src.shared.consts import VerifierType, LearnerType
from src.shared.cegis_values import CegisConfig
from src.plots.plot_lyap import plot_lyce


def main():
    batch_size = 1000
    system = partial(prajna07_modified, batch_size)
    activations = [
                    ActivationType.TANH,
                   ]
    hidden_neurons = [20]*len(activations)
    opts = {
        CegisConfig.N_VARS.k: 2,
        CegisConfig.LEARNER.k: LearnerType.NN,
        CegisConfig.VERIFIER.k: VerifierType.DREAL,
        CegisConfig.ACTIVATION.k: activations,
        CegisConfig.SYSTEM.k: system,
        CegisConfig.N_HIDDEN_NEURONS.k: hidden_neurons,
        CegisConfig.SP_SIMPLIFY.k: False,
        CegisConfig.SP_HANDLE.k: False,
        CegisConfig.SYMMETRIC_BELT.k: False,
    }
    try:
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

    except Exception as _:
        print(traceback.format_exc())


if __name__ == '__main__':
    torch.manual_seed(167)
    main()
