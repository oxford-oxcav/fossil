# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
import numpy
import torch
import timeit
from src.cegis import Cegis
from experiments.benchmarks.benchmark_ctrl import quadrotor2d_ctrl


from src.consts import *
from src.plots.plot_lyap import plot_lyce
import numpy as np


def test_lnn():
    #############################
    # converges in 10 sec
    #############################

    # using trajectory control
    benchmark = quadrotor2d_ctrl
    n_vars = 6
    system = benchmark

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [4] * len(activations)

    # control params
    n_ctrl_inputs = 2

    start = timeit.default_timer()
    opts = CegisConfig(
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.BARRIER,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        SYSTEM=system,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        SYMMETRIC_BELT=True,
        CTRLAYER=[5, n_ctrl_inputs],
        CTRLACTIVATION=[ActivationType.LINEAR],
    )
    c = Cegis(opts)
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
