import traceback
from functools import partial

import torch
import timeit

from experiments.benchmarks.benchmarks_bc import obstacle_avoidance
from src.barrier.cegis_barrier import Cegis
from src.shared.activations import ActivationType
from src.shared.cegis_values import CegisConfig
from src.shared.consts import VerifierType, LearnerType, TrajectoriserType, RegulariserType
from src.shared.cegis_values import CegisConfig


def main():

    batch_size = 500
    system = partial(obstacle_avoidance, batch_size)
    activations = [ActivationType.TANH]
    hidden_neurons = [10]
    try:
        start = timeit.default_timer()
        opts = {
            CegisConfig.N_VARS.k: 3,
            CegisConfig.LEARNER.k: LearnerType.NN,
            CegisConfig.VERIFIER.k: VerifierType.DREAL,
            CegisConfig.TRAJECTORISER.k: TrajectoriserType.DEFAULT,
            CegisConfig.REGULARISER.k: RegulariserType.DEFAULT,
            CegisConfig.ACTIVATION.k: activations,
            CegisConfig.SYSTEM.k: system,
            CegisConfig.N_HIDDEN_NEURONS.k: hidden_neurons,
            CegisConfig.SP_SIMPLIFY.k: False,
            CegisConfig.SP_HANDLE.k: False,
        }
        c = Cegis(**opts)
        state, _, __, ___ = c.solve()
        end = timeit.default_timer()

        print('Elapsed Time: {}'.format(end - start))
        print("Found? {}".format(state['found']))
    except Exception as _:
        print(traceback.format_exc())


if __name__ == '__main__':
    torch.manual_seed(167)
    main()
