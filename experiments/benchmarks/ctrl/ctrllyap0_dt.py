# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
import numpy
import torch
import timeit
from src.shared.components.cegis import Cegis
from experiments.benchmarks.benchmark_ctrl import trivial_ctrllyap


from src.shared.consts import *
from src.plots.plot_lyap import plot_lyce
import numpy as np


def test_lnn():
    # TEST for Control Lyapunov
    # pass the ctrl parameters from here (i.e. the main)
    benchmark = trivial_ctrllyap
    n_vars = 2
    system = benchmark

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [4] * len(activations)

    start = timeit.default_timer()
    opts = {
        CegisConfig.N_VARS.k: n_vars,
        CegisConfig.CERTIFICATE.k: CertificateType.LYAPUNOV,
        CegisConfig.LLO.k: False,
        CegisConfig.TIME_DOMAIN.k: TimeDomain.DISCRETE,
        CegisConfig.VERIFIER.k: VerifierType.DREAL,
        CegisConfig.ACTIVATION.k: activations,
        CegisConfig.SYSTEM.k: system,
        CegisConfig.N_HIDDEN_NEURONS.k: n_hidden_neurons,
        CegisConfig.CTRLAYER.k: [15, 2],
        CegisConfig.CTRLACTIVATION.k: [ActivationType.LINEAR],
    }
    c = Cegis(**opts)
    state, vars, f, iters = c.solve()
    stop = timeit.default_timer()
    print("Elapsed Time: {}".format(stop - start))

    # plotting -- only for 2-d systems
    if len(vars) == 2:
        plot_lyce(
            np.array(vars), state[CegisStateKeys.V], state[CegisStateKeys.V_dot], f
        )


if __name__ == "__main__":
    torch.manual_seed(169)
    test_lnn()
