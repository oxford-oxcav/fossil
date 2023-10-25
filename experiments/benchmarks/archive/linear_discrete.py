# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
import torch
import timeit
import numpy as np

from fossil.cegis import Cegis
from experiments.benchmarks.benchmarks_lyap import linear_discrete


from fossil.consts import *
from functools import partial
from fossil.plots.plot_lyap import plot_lyce_discrete


def test_lnn():
    n_vars = 2
    system = linear_discrete

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [5] * len(activations)

    opts = CegisConfig(
        N_VARS=n_vars,
        TIME_DOMAIN=TimeDomain.DISCRETE,
        VERIFIER=VerifierType.DREAL,
        CERTIFICATE=CertificateType.LYAPUNOV,
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

    # plotting -- only for 2-d systems
    if len(vars) == 2 and state[CegisStateKeys.found]:
        plot_lyce_discrete(
            np.array(vars),
            state[CegisStateKeys.V],
            state[CegisStateKeys.V_dot],
            f_learner,
        )


if __name__ == "__main__":
    torch.manual_seed(167)
    test_lnn()
