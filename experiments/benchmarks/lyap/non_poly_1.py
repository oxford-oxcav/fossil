# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
from src.shared.consts import *
from experiments.benchmarks.benchmarks_lyap import *


from src.plots.plot_lyap import plot_lyce
from src.shared.components.cegis import Cegis
from src.shared.utils import check_sympy_expression
from functools import partial

import numpy as np
import timeit


def test_lnn():

    n_vars = 2
    system = nonpoly1

    # define domain constraints
    inner_radius = 0.01

    # define NN parameters
    activations = [ActivationType.LINEAR, ActivationType.SQUARE]
    n_hidden_neurons = [20] * len(activations)

    opts = {
        CegisConfig.N_VARS.k: n_vars,
        CegisConfig.CERTIFICATE.k: CertificateType.LYAPUNOV,
        CegisConfig.TIME_DOMAIN.k: TimeDomain.CONTINUOUS,
        CegisConfig.VERIFIER.k: VerifierType.Z3,
        CegisConfig.ACTIVATION.k: activations,
        CegisConfig.SYSTEM.k: system,
        CegisConfig.N_HIDDEN_NEURONS.k: n_hidden_neurons,
        CegisConfig.LLO.k: True,
    }
    start = timeit.default_timer()
    c = Cegis(**opts)
    state, vars, f_learner, iters = c.solve()
    stop = timeit.default_timer()
    print("Elapsed Time: {}".format(stop - start))


if __name__ == "__main__":
    torch.manual_seed(167)
    test_lnn()
