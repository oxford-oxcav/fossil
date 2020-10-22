from experiments.benchmarks.benchmarks_bc import darboux
from src.barrier.cegis_barrier import Cegis
from src.shared.activations import ActivationType
from src.shared.cegis_values import CegisConfig
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
    system = partial(darboux, batch_size)
    activations = [ActivationType.LINEAR, ActivationType.LIN_TO_CUBIC, ActivationType.LINEAR]
    hidden_neurons = [10] * len(activations)
    try:
        start = timeit.default_timer()
        opts = {
            CegisConfig.N_VARS.k: 2,
            CegisConfig.LEARNER.k: LearnerType.NN,
            CegisConfig.VERIFIER.k: VerifierType.DREAL,
            CegisConfig.ACTIVATION.k: activations,
            CegisConfig.SYSTEM.k: system,
            CegisConfig.N_HIDDEN_NEURONS.k: hidden_neurons,
            CegisConfig.SP_SIMPLIFY.k: True,
        }
        c = Cegis(**opts)
        state, vars, f_learner, iters = c.solve()
        end = timeit.default_timer()

        # plotting -- only for 2-d systems
        if len(vars) == 2 and state['found']:
            plot_lyce(np.array(vars), state['V'],
                          state['V_dot'], f_learner)

        print('Elapsed Time: {}'.format(end - start))
        print("Found? {}".format(state['found']))
    except Exception as _:
        print(traceback.format_exc())


if __name__ == '__main__':
    torch.manual_seed(167)
    main()
