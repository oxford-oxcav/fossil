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
from experiments.benchmarks.models import NonPoly0
from src.cegis import Cegis
from src.consts import *


def test_lnn():
    ###########################################
    ### Converges in 2.5 s on 8th loop
    ### Trivial example
    ###
    #############################################
    n_vars = 2

    system = NonPoly0()
    batch_size = 500

    XD = sets.Sphere([0, 0], 1.1)
    XS = sets.Sphere([-0.3, -0.3], 0.6)
    XI = sets.Sphere([-0.6, -0.6], 0.01)
    XG = sets.Sphere([0, 0], 0.1)

    SU = sets.SetMinus(XD, XS)  # Data for unsafe set
    SD = sets.SetMinus(XS, XG)  # Data for lie set

    D = {
        "lie": XD,
        "init": XI,
        "safe": XS,
        "goal": XG,
    }
    symbolic_domains = {
        "lie": XD.generate_domain,
        "init": XI.generate_domain,
        "safe_border": XS.generate_boundary,
        "safe": XS.generate_domain,
        "goal": XG.generate_domain,
    }
    data = {
        "lie": SD.generate_data(batch_size),
        "init": XI.generate_data(100),
        "unsafe": SU.generate_data(1000),
    }
    F = lambda *args: (system, symbolic_domains, data, sets.inf_bounds_n(2))

    # plot_benchmark_plane(
    #     system,
    #     D,
    #     xrange=[-1.1, 1.1],
    #     yrange=[-1.1, 1.1],
    # )

    # define NN parameters
    activations = [ActivationType.LIN_TO_QUARTIC]
    n_hidden_neurons = [16] * len(activations)

    opts = CegisConfig(
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.RWS,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        SYSTEM=F,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        CEGIS_MAX_ITERS=10,
    )

    start = timeit.default_timer()
    c = Cegis(opts)
    c.solve()
    stop = timeit.default_timer()
    print("Elapsed Time: {}".format(stop - start))

    plotting.benchmark(
        system,
        c.learner,
        domains=D,
        levels=[0],
        xrange=[-1.1, 1.1],
        yrange=[-1.1, 1.1],
    )


if __name__ == "__main__":
    torch.manual_seed(167)
    test_lnn()
