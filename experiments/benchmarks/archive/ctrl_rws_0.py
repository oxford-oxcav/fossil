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
from fossil.cegis import Cegis
from fossil.consts import *


def test_lnn():
    ###########################################
    ### Converes in 6s on 14th loop
    ### This is super brittle with control. The dynamics often just become unstable and messy.
    ### I've added a term to the loss to try to encourage stability, but the balance of the hyperparameters is tricky.
    #############################################
    n_vars = 2

    ol_system = models.Benchmark1()
    system = lambda ctrl: models.GeneralClosedLoopModel(ol_system, ctrl)

    batch_size = 500

    XD = sets.Sphere([0, 0], 1.5)
    XS = sets.Sphere([-0.3, -0.3], 0.7)
    XI = sets.Sphere([-0.6, -0.6], 0.01)
    XG = sets.Sphere([0, 0], 0.1)

    SU = sets.SetMinus(XD, XS)  # Data for unsafe set
    SD = sets.SetMinus(XS, XG)  # Data for lie set (domain less unsafe and goal)

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
        "lie": SD.generate_data(100),
        "init": XI.generate_data(100),
        "unsafe": SU.generate_data(100),
    }
    F = lambda *args: (system(*args), symbolic_domains, data, sets.inf_bounds_n(2))

    # plot_benchmark_plane(
    #     system,
    #     D,
    #     xrange=[-1.1, 1.1],
    #     yrange=[-1.1, 1.1],
    # )

    # define NN parameters
    activations = [ActivationType.POLY_4]
    n_hidden_neurons = [10] * len(activations)

    opts = CegisConfig(
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.RWS,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        SYSTEM=F,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        CEGIS_MAX_ITERS=15,
        CTRLAYER=[5, 2],
        CTRLACTIVATION=[ActivationType.LINEAR],
    )

    start = timeit.default_timer()
    c = Cegis(opts)
    c.solve()
    stop = timeit.default_timer()
    print("Elapsed Time: {}".format(stop - start))

    plotting.benchmark(
        c.f,
        c.learner,
        D,
        xrange=[-1.5, 1.5],
        yrange=[-1.5, 1.5],
        levels=[0],
    )


if __name__ == "__main__":
    torch.manual_seed(167)
    test_lnn()
