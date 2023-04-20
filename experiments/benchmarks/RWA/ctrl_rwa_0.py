# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
import torch
import timeit


from src.shared.components.cegis import Cegis
import experiments.benchmarks.models as models
import experiments.benchmarks.domain_fcns as sets
from src.shared.consts import *
import src.plots.plot_fcns as plotting


def test_lnn():
    ###########################################
    ### DOES NOT WORK
    ### MODEL IS CLEARLY INCORRECT.
    ### Think this is due to incorrect certificate conditions from safe and unsafe discrepancy (will talk)
    #############################################
    n_vars = 2

    ol_system = models.Benchmark1()
    system = lambda ctrl: models.GeneralClosedLoopModel(ol_system, ctrl)
    batch_size = 500

    XD = sets.Sphere([0, 0], 1.1)

    # XU = sets.SetMinus(sets.Rectangle([0, 0], [1.2, 1.2]), sets.Sphere([0.6, 0.6], 0.4))
    XU = sets.Sphere([0.4, 0.4], 0.1)
    XI = sets.Sphere([-0.6, -0.6], 0.01)

    # XU = sets.SetMinus(sets.Rectangle([0, 0], [1.2, 1.2]), sets.Sphere([0.6, 0.6], 0.4))
    XG = sets.Sphere([0, 0], 0.1)
    SD = sets.SetMinus(sets.SetMinus(XD, XG), XU)

    D = {
        "lie": XD,
        "init": XI,
        "unsafe": XU,
        "goal": XG,
    }
    symbolic_domains = {
        "lie": XD.generate_domain,
        "init": XI.generate_domain,
        "unsafe_border": XU.generate_boundary,
        "unsafe": XU.generate_interior,
        "goal": XG.generate_domain,
    }
    data = {
        "lie": SD.generate_data(batch_size),
        "init": XI.generate_data(100),
        "unsafe": XU.generate_data(100),
    }
    F = lambda ctrl: (system(ctrl), symbolic_domains, data, sets.inf_bounds_n(2))

    # plot_benchmark_plane(
    #     system,
    #     D,
    #     xrange=[-1.1, 1.1],
    #     yrange=[-1.1, 1.1],
    # )

    # define NN parameters
    activations = [ActivationType.LIN_TO_QUARTIC]
    n_hidden_neurons = [10] * len(activations)

    opts = CegisConfig(
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.RWA,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        SYSTEM=F,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        CEGIS_MAX_ITERS=10,
        CTRLAYER=[15, 2],
        CTRLACTIVATION=[ActivationType.LINEAR],
    )

    start = timeit.default_timer()
    c = Cegis(opts)
    state, vars, f, iters = c.solve()
    stop = timeit.default_timer()
    print("Elapsed Time: {}".format(stop - start))

    plotting.benchmark(
        f,
        c.learner,
        D,
        xrange=[-1.1, 1.1],
        yrange=[-1.1, 1.1],
        levels=[0],
    )


if __name__ == "__main__":
    torch.manual_seed(167)
    test_lnn()
