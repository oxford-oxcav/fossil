# pylint: disable=not-callable
from experiments.benchmarks.benchmarks_bc import hi_ord_4
from src.shared.consts import VerifierType, LearnerType, TrajectoriserType, RegulariserType
from src.shared.activations import ActivationType
from src.shared.cegis_values import CegisConfig, CegisStateKeys
from src.barrier.cegis_barrier import Cegis
from functools import partial
import traceback
import timeit
import torch


def main():

    batch_size = 1000
    system = partial(hi_ord_4, batch_size)
    activations = [ActivationType.LINEAR]
    hidden_neurons = [20]

    opts = {
        CegisConfig.N_VARS.k: 4,
        CegisConfig.LEARNER.k: LearnerType.NN,
        CegisConfig.VERIFIER.k: VerifierType.DREAL,
        CegisConfig.TRAJECTORISER.k: TrajectoriserType.DEFAULT,
        CegisConfig.REGULARISER.k: RegulariserType.DEFAULT,
        CegisConfig.ACTIVATION.k: activations,
        CegisConfig.SYSTEM.k: system,
        CegisConfig.N_HIDDEN_NEURONS.k: hidden_neurons,
        CegisConfig.SYMMETRIC_BELT.k: False,
        CegisConfig.SP_HANDLE.k: True,
        CegisConfig.SP_SIMPLIFY.k: True,
        CegisConfig.ROUNDING.k: 2
    }

    start = timeit.default_timer()
    c = Cegis(**opts)
    state, vars, f_learner, iters = c.solve()
    end = timeit.default_timer()

    print('Elapsed Time: {}'.format(end - start))
    print("Found? {}".format(state[CegisStateKeys.found]))


if __name__ == '__main__':
    torch.manual_seed(167)
    main()

