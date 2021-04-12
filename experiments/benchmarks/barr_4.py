# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
# pylint: disable=not-callable
import traceback
from functools import partial

import torch
import numpy as np
import timeit

from experiments.benchmarks.benchmarks_bc import obstacle_avoidance as barr_4
from src.barrier.cegis_barrier import Cegis
from src.shared.activations import ActivationType
from src.shared.consts import VerifierType, LearnerType, ConsolidatorType, TranslatorType
from src.shared.cegis_values import CegisConfig, CegisStateKeys


def main():
    batch_size = 2000
    system = partial(barr_4, batch_size)
    activations = [
                    ActivationType.LIN_TO_CUBIC,
                   ]
    hidden_neurons = [5]*len(activations)
    opts = {
        CegisConfig.N_VARS.k: 3,
        CegisConfig.ACTIVATION.k: activations,
        CegisConfig.LEARNER.k: LearnerType.NN,
        CegisConfig.VERIFIER.k: VerifierType.DREAL,
        CegisConfig.CONSOLIDATOR.k: ConsolidatorType.DEFAULT,
        CegisConfig.TRANSLATOR.k: TranslatorType.DEFAULT,
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


if __name__ == '__main__':
    torch.manual_seed(167)
    main()
