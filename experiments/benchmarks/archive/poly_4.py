# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
import torch
import timeit
from fossil.cegis import Cegis
from experiments.benchmarks.benchmarks_lyap import *


from fossil.consts import *
from functools import partial
from fossil.plots.plot_lyap import plot_lyce


def test_lnn():
    n_vars = 2
    system = poly_4

    # define domain constraints
    inner_radius = 0.01

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [5] * len(activations)

    opts = CegisConfig(
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.LYAPUNOV,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        SYSTEM=system,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        LLO=True,
    )

    start = timeit.default_timer()
    c = Cegis(opts)
    state, vars, f_learner, iters = c.solve()
    stop = timeit.default_timer()
    print("Elapsed Time: {}".format(stop - start))


if __name__ == "__main__":
    torch.manual_seed(167)
    test_lnn()
