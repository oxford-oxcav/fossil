# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
import numpy
import torch
import timeit
from fossil.cegis import Cegis
from experiments.benchmarks.benchmark_ctrl import ctrlbarr_car


from fossil.consts import *
from fossil.plots.plot_lyap import plot_lyce
import numpy as np


def test_lnn():
    # TEST for Control Lyapunov
    # pass the ctrl parameters from here (i.e. the main)
    benchmark = ctrlbarr_car
    n_vars = 3
    system = benchmark

    # define NN parameters
    barr_activations = [ActivationType.SQUARE]
    barr_hidden_neurons = [5] * len(barr_activations)

    # ctrl params
    n_ctrl_inputs = 3

    start = timeit.default_timer()
    opts = CegisConfig(
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.BARRIER,
        LLO=False,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=barr_activations,
        SYSTEM=system,
        N_HIDDEN_NEURONS=barr_hidden_neurons,
        CTRLAYER=[25, n_ctrl_inputs],
        CTRLACTIVATION=[ActivationType.LINEAR],
    )
    c = Cegis(opts)
    state, vars, f, iters = c.solve()
    stop = timeit.default_timer()
    print("Elapsed Time: {}".format(stop - start))

    # plotting -- only for 2-d systems
    if len(vars) == 2:
        plot_lyce(
            np.array(vars), state[CegisStateKeys.V], state[CegisStateKeys.V_dot], f
        )


if __name__ == "__main__":
    torch.manual_seed(167)
    test_lnn()
