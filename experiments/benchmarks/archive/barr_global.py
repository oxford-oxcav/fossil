# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import timeit

# pylint: disable=not-callable
import torch

import fossil.domains as sets
from experiments.benchmarks.models import Linear0
from fossil.cegis import Cegis
from fossil.consts import *
from fossil.plotting import benchmark_plane


def test_lnn():
    #######################################
    ### Works on second iter almost immediately, for Z3 and DReal
    ### This is a proof of concept. I don't know how well it will extend to other systems.
    #######################################
    n_vars = 2
    system = Linear0()
    XI = sets.Sphere([0.2, 0.2], 0.1)
    XU = sets.Sphere([1, 1], 0.1)
    SD = sets.Torus([0, 0], 100.0, 0.0001)
    XD = sets.Reals(2)
    D = {"init": XI, "unsafe": XU}
    domains = {
        "lie": XD.generate_domain,
        "init": XI.generate_domain,
        "unsafe": XU.generate_domain,
    }
    data = {
        "lie": SD.generate_data(100),
        "init": XI.generate_data(200),
        "unsafe": XU.generate_data(200),
    }
    F = lambda *args: (system, domains, data, sets.inf_bounds_n(2))

    # define NN parameters
    activations = (ActivationType.SQUARE,)
    n_hidden_neurons = (5,) * len(activations)

    opts = CegisConfig(
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.BARRIER,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        SYSTEM=F,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        CEGIS_MAX_ITERS=2,
    )

    start = timeit.default_timer()
    c = Cegis(opts)
    c.solve()
    stop = timeit.default_timer()
    print("Elapsed Time: {}".format(stop - start))

    benchmark_plane(
        c.f,
        D,
        c.learner,
        levels=[0],
    )


if __name__ == "__main__":
    torch.manual_seed(167)
    test_lnn()
