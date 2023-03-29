# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
from experiments.benchmarks.benchmarks_bc import hi_ord_6
from src.shared.consts import *


from src.shared.components.cegis import Cegis
from functools import partial
import timeit
import torch


def main():

    system = hi_ord_6
    activations = [ActivationType.LINEAR]
    hidden_neurons = [10]
    opts = {
        CegisConfig.N_VARS.k: 6,
        CegisConfig.CERTIFICATE.k: CertificateType.BARRIER,
        CegisConfig.TIME_DOMAIN.k: TimeDomain.CONTINUOUS,
        CegisConfig.VERIFIER.k: VerifierType.DREAL,
        CegisConfig.ACTIVATION.k: activations,
        CegisConfig.SYSTEM.k: system,
        CegisConfig.N_HIDDEN_NEURONS.k: hidden_neurons,
        CegisConfig.SYMMETRIC_BELT.k: False,
        CegisConfig.ROUNDING.k: 2,
    }

    start = timeit.default_timer()
    c = Cegis(**opts)
    state, _, __, ___ = c.solve()
    end = timeit.default_timer()

    print("Elapsed Time: {}".format(end - start))
    print("Found? {}".format(state[CegisStateKeys.found]))


if __name__ == "__main__":
    torch.manual_seed(167)
    main()
