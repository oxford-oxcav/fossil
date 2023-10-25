# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
from experiments.benchmarks.benchmarks_bc import twod_hybrid
from fossil.cegis import Cegis

from fossil.consts import *


from functools import partial
import timeit
import torch


def main():
    system = twod_hybrid
    activations = [ActivationType.POLY_2]
    hidden_neurons = [3] * len(activations)
    start = timeit.default_timer()
    opts = CegisConfig(
        N_VARS=2,
        CERTIFICATE=CertificateType.BARRIER,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.Z3,
        ACTIVATION=activations,
        SYSTEM=system,
        N_HIDDEN_NEURONS=hidden_neurons,
        SYMMETRIC_BELT=False,
    )
    c = Cegis(opts)
    state, vars, f_learner, iters = c.solve()
    end = timeit.default_timer()

    print("Elapsed Time: {}".format(end - start))
    print("Found? {}".format(state[CegisStateKeys.found]))


if __name__ == "__main__":
    torch.manual_seed(167)
    main()
