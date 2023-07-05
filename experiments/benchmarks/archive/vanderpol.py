# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import timeit

# pylint: disable=not-callable
import torch

import src.domains as sets
import src.plotting as plotting
from experiments.benchmarks.models import VanDerPol
from src.cegis import Cegis
from src.consts import *


def test_lnn():
    ###########################################
    ### Converges in 0.1s in second step
    ### To be improved. Just to demonstrate/ experiment with plotting functionality
    ###
    #############################################
    n_vars = 2

    system = VanDerPol()
    batch_size = 200

    # XU = sets.SetMinus(sets.Rectangle([0, 0], [1.2, 1.2]), sets.Sphere([0.6, 0.6], 0.4))

    XD = sets.Rectangle([-3, -3], [3, 3])
    XU = sets.Torus([0, 0], 3, 2.5)
    # XU = sets.SetMinus(XD, sets.Rectangle([-2.5, -2.5], [2.5, 2.5]))
    XI = sets.Sphere([0, 0], 0.5)

    D = {"lie": XD, "unsafe": XU, "init": XI}
    # plotting.benchmark_plane(system, D, xrange=[-3, 3], yrange=[-3, 3])
    # plotting.show()
    domains = {lab: dom.generate_domain for lab, dom in D.items()}
    data = data = {
        "lie": XD.generate_data(1000),
        "init": XI.generate_data(100),
        "unsafe": XU.generate_data(100),
    }
    F = lambda *args: (system, domains, data, sets.inf_bounds_n(2))

    # define NN parameters
    activations = [ActivationType.TANH]
    n_hidden_neurons = [16] * len(activations)

    opts = CegisConfig(
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.BARRIER,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        SYSTEM=F,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        SYMMETRIC_BELT=False,
        CEGIS_MAX_ITERS=15,
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
        xrange=[-5, 5],
        yrange=[-5, 5],
        levels=[0],
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.manual_seed(167)
    test_lnn()