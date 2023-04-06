# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
import torch
import numpy as np
import timeit
from src.shared.components.cegis import Cegis
from experiments.benchmarks.benchmarks_rws import rws_linear


from src.shared.consts import *
from src.plots.plot_lyap import plot_lyce


def test_lnn():
    benchmark = rws_linear
    n_vars = 2
    n_ctrl_inputs = 2
    system = benchmark

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [5] * len(activations)

    start = timeit.default_timer()
    opts = CegisConfig(
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.CTRLRWS,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        SYSTEM=system,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        CTRLAYER=[10, n_ctrl_inputs],
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
    torch.manual_seed(167)
    torch.set_num_threads(1)
    test_lnn()
