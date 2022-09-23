# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable

import torch
import timeit

from experiments.benchmarks.benchmarks_bc import obstacle_avoidance as barr_4
from src.shared.components.cegis import Cegis
from src.shared.activations import ActivationType
from src.shared.consts import VerifierType, TimeDomain, CertificateType
from src.shared.cegis_values import CegisConfig, CegisStateKeys


def main():
    system = barr_4
    activations = [ActivationType.LIN_TO_CUBIC]
    hidden_neurons = [25]
    opts = {
        CegisConfig.N_VARS.k: 3,
        CegisConfig.CERTIFICATE.k: CertificateType.BARRIER,
        CegisConfig.ACTIVATION.k: activations,
        CegisConfig.TIME_DOMAIN.k: TimeDomain.CONTINUOUS,
        CegisConfig.VERIFIER.k: VerifierType.DREAL,
        CegisConfig.SYSTEM.k: system,
        CegisConfig.N_HIDDEN_NEURONS.k: hidden_neurons,
    }

    start = timeit.default_timer()
    c = Cegis(**opts)
    state, vars, f, iters = c.solve()
    end = timeit.default_timer()

    print("Elapsed Time: {}".format(end - start))
    print("Found? {}".format(state[CegisStateKeys.found]))


if __name__ == "__main__":
    torch.manual_seed(167)
    main()
