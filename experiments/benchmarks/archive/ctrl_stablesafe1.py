# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import timeit

# pylint: disable=not-callable
import torch

import experiments.benchmarks.models as models
import fossil.domains as sets
import fossil.plotting as plotting
from fossil.cegis import DoubleCegis
from fossil.consts import *
from fossil.plots.plot_lyap import plot_lyce


def test_lnn():
    ###########################################
    ### Converges in 1.6s in second step
    ### Trivial example
    ### Currently DoubleCegis does not work with consolidator
    #############################################
    n_vars = 2

    ol_system = models.Benchmark1()
    system = lambda ctrl: models.GeneralClosedLoopModel(ol_system, ctrl)
    batch_size = 500

    XD = sets.Torus([0, 0], 1.1, 0.01)

    # XU = sets.SetMinus(sets.Rectangle([0, 0], [1.2, 1.2]), sets.Sphere([0.6, 0.6], 0.4))
    XU = sets.Sphere([0.4, 0.4], 0.1)
    XI = sets.Sphere([-0.6, -0.6], 0.01)
    D = {
        "lie": XD,
        "init": XI,
        "unsafe": XU,
    }
    domains = {lab: dom.generate_domain for lab, dom in D.items()}
    data = {lab: dom.generate_data(batch_size) for lab, dom in D.items()}
    F = lambda ctrl: (system(ctrl), domains, data, sets.inf_bounds_n(2))

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [12] * len(activations)

    opts = CegisConfig(
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.STABLESAFE,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        SYSTEM=F,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        SYMMETRIC_BELT=False,
        CTRLAYER=[15, 2],
        CTRLACTIVATION=[ActivationType.LINEAR],
    )

    start = timeit.default_timer()
    c = DoubleCegis(opts)
    state, vars, f, iters = c.solve()
    stop = timeit.default_timer()
    print("Elapsed Time: {}".format(stop - start))

    if len(vars) == 3:
        plot_lyce(
            np.array(vars),
            state[CegisStateKeys.V][0],
            state[CegisStateKeys.V_dot][0],
            f,
        )

    plotting.benchmark(
        f,
        c.lyap_learner,
        D,
        xrange=[-1.1, 1.1],
        yrange=[-1.1, 1.1],
        levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
    )


if __name__ == "__main__":
    torch.manual_seed(167)
    test_lnn()
