# pylint: disable=not-callable
import traceback
from functools import partial

import torch
import numpy as np
import timeit

from experiments.benchmarks.benchmarks_bc import barr_3
from src.barrier.cegis_barrier import Cegis
from src.shared.activations import ActivationType
from src.shared.consts import VerifierType, LearnerType, ConsolidatorType, TranslatorType
from src.shared.cegis_values import CegisConfig, CegisStateKeys
from src.plots.plot_barriers import plot_pjmod_bench


def main():
    batch_size = 2000
    system = partial(barr_3, batch_size)
    activations = [
                    ActivationType.SIGMOID, ActivationType.SIGMOID
                   ]
    hidden_neurons = [20]*len(activations)
    opts = {
        CegisConfig.N_VARS.k: 2,
        CegisConfig.LEARNER.k: LearnerType.NN,
        CegisConfig.VERIFIER.k: VerifierType.DREAL,
        CegisConfig.CONSOLIDATOR.k: ConsolidatorType.DEFAULT,
        CegisConfig.TRANSLATOR.k: TranslatorType.DEFAULT,
        CegisConfig.ACTIVATION.k: activations,
        CegisConfig.SYSTEM.k: system,
        CegisConfig.N_HIDDEN_NEURONS.k: hidden_neurons,
        CegisConfig.SP_SIMPLIFY.k: False,
        CegisConfig.SP_HANDLE.k: False,
    }

    start = timeit.default_timer()
    c = Cegis(**opts)
    state, vars, f, iters = c.solve()
    end = timeit.default_timer()

    print('Elapsed Time: {}'.format(end - start))
    print("Found? {}".format(state[CegisStateKeys.found]))

    # plotting -- only for 2-d systems
    if state[CegisStateKeys.found]:
        plot_pjmod_bench(np.array(vars), state[CegisStateKeys.V])


if __name__ == '__main__':
    torch.manual_seed(167)
    main()
