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
from fossil.utils import check_sympy_expression


def test_lnn():
    n_vars = 2
    system = poly_2

    # define NN parameters
    activations = [ActivationType.COSH]
    n_hidden_neurons = [10] * len(activations)

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

    # plotting -- only for 2-d systems
    if len(vars) == 2 and state[CegisStateKeys.found]:
        V, Vdot = check_sympy_expression(state, system)
        plot_lyce(np.array(vars), V, Vdot, f_learner)


if __name__ == "__main__":
    torch.manual_seed(167)
    test_lnn()
