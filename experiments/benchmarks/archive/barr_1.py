# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import timeit

import numpy as np
import torch

import fossil.plotting as plotting

# pylint: disable=not-callable
from experiments.benchmarks.benchmarks_bc import barr_1
from fossil.cegis import Cegis
from fossil.consts import *


def main():
    system = barr_1
    activations = [ActivationType.LINEAR]
    hidden_neurons = [10] * len(activations)
    start = timeit.default_timer()
    opts = CegisConfig(
        N_VARS=2,
        CERTIFICATE=CertificateType.BARRIER,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        SYSTEM=system,
        N_HIDDEN_NEURONS=hidden_neurons,
        SYMMETRIC_BELT=True,
    )
    c = Cegis(opts)
    state, vars, f_learner, iters = c.solve()
    end = timeit.default_timer()
    print("Elapsed Time: {}".format(end - start))
    print("Found? {}".format(state[CegisStateKeys.found]))

    # plotting -- only for 2-d systems
    plotting.benchmark(
        c.f,
        c.learner,
        {},
    )


if __name__ == "__main__":
    torch.manual_seed(167)
    main()
