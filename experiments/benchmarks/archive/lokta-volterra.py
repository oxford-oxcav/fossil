# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import timeit

# pylint: disable=not-callable
import torch

import fossil.domains as sets
import fossil.plotting as plotting
from experiments.benchmarks.models import LoktaVolterra
from fossil.cegis import Cegis
from fossil.consts import *


def test_lnn():
    ###########################################
    ### Converges in 0.1s in second step
    ### To be improved. Just to demonstrate/ experiment with plotting functionality
    ###
    #############################################
    n_vars = 2

    system = LoktaVolterra()
    batch_size = 1

    XD = sets.Rectangle([0, 0], [0.7, 0.7])

    # XU = sets.SetMinus(sets.Rectangle([0, 0], [1.2, 1.2]), sets.Sphere([0.6, 0.6], 0.4))

    XU = sets.Sphere([0.2, 0.2], 0.1)
    XI = sets.Sphere([0.6, 0.6], 0.01)
    D = {"lie": XD, "unsafe": XU, "init": XI}
    domains = {lab: dom.generate_domain for lab, dom in D.items()}
    data = {lab: dom.generate_data(batch_size) for lab, dom in D.items()}
    F = lambda *args: (system, domains, data, sets.inf_bounds_n(2))

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [12] * len(activations)

    opts = CegisConfig(
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.BARRIER,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        SYSTEM=F,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        SYMMETRIC_BELT=True,
    )

    start = timeit.default_timer()
    c = Cegis(opts)
    c.solve()
    stop = timeit.default_timer()
    print("Elapsed Time: {}".format(stop - start))

    plotting.benchmark(
        system,
        c.learner,
        D,
        xrange=[0, 0.7],
        yrange=[0, 0.7],
        levels=[0],
    )


if __name__ == "__main__":
    torch.manual_seed(167)
    test_lnn()
