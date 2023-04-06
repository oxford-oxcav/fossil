# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
import torch
import timeit


from src.shared.components.cegis import DoubleCegis
from experiments.benchmarks.models import NonPoly0
import experiments.benchmarks.domain_fcns as sets
from src.shared.consts import *
import src.plots.plot_fcns as plotting


def test_lnn():
    ###########################################
    ### Converges in 1.6s in second step
    ### Trivial example
    ### Currently DoubleCegis does not work with consolidator
    #############################################
    n_vars = 2

    system = NonPoly0()
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
    F = lambda *args: (system, domains, data, sets.inf_bounds_n(2))

    # define NN parameters
    activations = [ActivationType.SQUARE]
    activations_alt = [ActivationType.TANH]
    n_hidden_neurons = [12] * len(activations)
    n_hidden_neurons_alt = [5] * len(activations_alt)

    opts = CegisConfig(
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.STABLESAFE,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        ACTIVATION_ALT=activations_alt,
        N_HIDDEN_NEURONS_ALT=n_hidden_neurons_alt,
        SYSTEM=F,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        SYMMETRIC_BELT=False,
    )

    start = timeit.default_timer()
    c = DoubleCegis(opts)
    c.solve()
    stop = timeit.default_timer()
    print("Elapsed Time: {}".format(stop - start))

    plotting.benchmark_plane(
        system,
        D,
        certificate=c.barr_learner,
        xrange=[-1.1, 1.1],
        yrange=[-1.1, 1.1],
        levels=[0],
    )

    plotting.benchmark_3d(
        c.barr_learner,
        D,
        xrange=[-1.1, 1.1],
        yrange=[-1.1, 1.1],
        levels=[0],
    )

    plotting.show()


if __name__ == "__main__":
    torch.manual_seed(167)
    test_lnn()
