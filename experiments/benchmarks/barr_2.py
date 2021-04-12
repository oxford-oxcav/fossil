# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
# pylint: disable=not-callable
import traceback
from functools import partial

import torch
import timeit

from experiments.benchmarks.benchmarks_bc import barr_2
from src.barrier.cegis_barrier import Cegis
from src.shared.activations import ActivationType
from src.shared.consts import VerifierType, LearnerType, ConsolidatorType, TranslatorType
from src.shared.cegis_values import CegisConfig, CegisStateKeys
from src.plots.plot_barriers import plot_exponential_bench
import numpy as np


def main():
    batch_size = 500
    system = partial(barr_2, batch_size)
    activations = [ActivationType.SOFTPLUS]
    hidden_neurons = [20]
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
        CegisConfig.SYMMETRIC_BELT.k: False,
    }

    start = timeit.default_timer()
    c = Cegis(**opts)
    state, vars, f, iters = c.solve()
    end = timeit.default_timer()

    print('Elapsed Time: {}'.format(end - start))
    print("Found? {}".format(state[CegisStateKeys.found]))

    # plotting -- only for 2-d systems
    if state[CegisStateKeys.found]:
        plot_exponential_bench(np.array(vars), state[CegisStateKeys.V])


if __name__ == '__main__':
    torch.manual_seed(167)
    main()
