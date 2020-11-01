# pylint: disable=not-callable
from experiments.benchmarks.benchmarks_bc import barr_1
from src.barrier.cegis_barrier import Cegis
from src.shared.activations import ActivationType
from src.shared.cegis_values import CegisConfig, CegisStateKeys
from src.shared.consts import VerifierType, LearnerType, TrajectoriserType, RegulariserType
from src.shared.activations import ActivationType
from src.barrier.cegis_barrier import Cegis
from src.plots.plot_barriers import plot_darboux_bench
from functools import partial
import numpy as np
import traceback
import timeit
import torch


def main():

    batch_size = 500
    system = partial(barr_1, batch_size)
    activations = [ActivationType.LINEAR, ActivationType.LIN_TO_CUBIC, ActivationType.LINEAR]
    hidden_neurons = [10] * len(activations)
    start = timeit.default_timer()
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
    }
    c = Cegis(**opts)
    state, vars, f_learner, iters = c.solve()
    end = timeit.default_timer()
    print('Elapsed Time: {}'.format(end - start))
    print("Found? {}".format(state[CegisStateKeys.found]))

    # plotting -- only for 2-d systems
    if state[CegisStateKeys.found]:
        plot_darboux_bench(np.array(vars), state[CegisStateKeys.V])


if __name__ == '__main__':
    torch.manual_seed(167)
    main()
