from experiments.benchmarks.benchmarks_bc import darboux
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
    activations = [ActivationType.LINEAR, ActivationType.LIN_SQUARE_CUBIC, ActivationType.LINEAR]
    hidden_neurons = [10] * len(activations)
    opts = {CegisConfig.SP_SIMPLIFY.k: True}
    try:
        start = timeit.default_timer()
        c = Cegis(2, LearnerType.NN, VerifierType.DREAL, activations, system, hidden_neurons,
                  **opts)
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
