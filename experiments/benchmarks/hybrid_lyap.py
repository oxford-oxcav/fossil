from experiments.benchmarks.benchmarks_lyap import twod_hybrid
from src.shared.activations import ActivationType
from src.shared.cegis_values import CegisConfig
from src.shared.consts import VerifierType, LearnerType
from src.shared.activations import ActivationType
from src.shared.cegis_values import CegisConfig
from src.lyap.cegis_lyap import Cegis
from src.plots.plot_lyap import plot_lyce

from functools import partial
import numpy as np
import traceback
import timeit
import torch


def main():

    batch_size = 1000
    system = partial(twod_hybrid, batch_size)
    activations = [ActivationType.SQUARE]
    hidden_neurons = [10] * len(activations)
    try:
        start = timeit.default_timer()
        opts = {
            CegisConfig.N_VARS.k: 2,
            CegisConfig.LEARNER.k: LearnerType.NN,
            CegisConfig.VERIFIER.k: VerifierType.Z3,
            CegisConfig.ACTIVATION.k: activations,
            CegisConfig.SYSTEM.k: system,
            CegisConfig.N_HIDDEN_NEURONS.k: hidden_neurons,
            CegisConfig.INNER_RADIUS.k: 0.0,
            CegisConfig.OUTER_RADIUS.k: 10.0,
            CegisConfig.SP_HANDLE.k: False,
            CegisConfig.SP_SIMPLIFY.k: False,
            CegisConfig.LLO.k: True,
        }
        c = Cegis(**opts)
        state, vars, f_learner, iters = c.solve()
        end = timeit.default_timer()

        print('Elapsed Time: {}'.format(end - start))
        print("Found? {}".format(state['found']))
    except Exception as _:
        print(traceback.format_exc())


if __name__ == '__main__':
    torch.manual_seed(167)
    main()