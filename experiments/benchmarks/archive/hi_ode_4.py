# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
from experiments.benchmarks.benchmarks_bc import hi_ord_4
from fossil.consts import *


from fossil.cegis import Cegis
from functools import partial
import traceback
import timeit
import torch


def main():
    system = hi_ord_4
    activations = [ActivationType.LINEAR]
    hidden_neurons = [1]

    opts = CegisConfig(
        N_VARS=4,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        CERTIFICATE=CertificateType.BARRIER,
        ACTIVATION=activations,
        SYSTEM=system,
        N_HIDDEN_NEURONS=hidden_neurons,
        SYMMETRIC_BELT=False,
    )

    start = timeit.default_timer()
    c = Cegis(opts)
    state, vars, f_learner, iters = c.solve()
    end = timeit.default_timer()

    print("Elapsed Time: {}".format(end - start))
    print("Found? {}".format(state[CegisStateKeys.found]))


if __name__ == "__main__":
    torch.manual_seed(167)
    main()
