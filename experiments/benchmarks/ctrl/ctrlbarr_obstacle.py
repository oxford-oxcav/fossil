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
from experiments.benchmarks.benchmark_ctrl import ctrl_obstacle_avoidance


from src.shared.consts import *
from src.plots.plot_lyap import plot_lyce
import numpy as np


def test_lnn():

    ###############################
    # This is a great idea!
    # takes 3.3 secs, at iter 3
    ###############################

    # using trajectory control
    benchmark = ctrl_obstacle_avoidance
    n_vars = 3
    system = benchmark

    # define NN parameters
    barr_activations = [ActivationType.TANH]
    barr_hidden_neurons = [15] * len(barr_activations)

    # ctrl params
    n_ctrl_inputs = 1

    start = timeit.default_timer()
    opts = {
        CegisConfig.N_VARS.k: n_vars,
        CegisConfig.CERTIFICATE.k: CertificateType.BARRIER,
        CegisConfig.TIME_DOMAIN.k: TimeDomain.CONTINUOUS,
        CegisConfig.VERIFIER.k: VerifierType.DREAL,
        CegisConfig.ACTIVATION.k: barr_activations,
        CegisConfig.SYSTEM.k: system,
        CegisConfig.N_HIDDEN_NEURONS.k: barr_hidden_neurons,
        CegisConfig.CTRLAYER.k: [20, n_ctrl_inputs],
        CegisConfig.CTRLACTIVATION.k: [ActivationType.LINEAR],
        CegisConfig.SYMMETRIC_BELT.k: True,
    }
    c = Cegis(**opts)
    state, vars, f, iters = c.solve()
    stop = timeit.default_timer()
    print("Elapsed Time: {}".format(stop - start))

    # plotting -- only for 2-d systems
    if len(vars) == 2 and state[CegisStateKeys.found]:
        plot_lyce(
            np.array(vars), state[CegisStateKeys.V], state[CegisStateKeys.V_dot], f
        )


if __name__ == "__main__":
    torch.manual_seed(169)
    test_lnn()
