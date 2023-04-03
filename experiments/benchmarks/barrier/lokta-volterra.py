# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
import torch
import timeit


from src.shared.components.cegis import Cegis
from experiments.benchmarks.models import LoktaVolterra
import experiments.benchmarks.domain_fcns as sets
from src.shared.consts import *
from src.plots.plot_fcns import plot_benchmark


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

    opts = {
        CegisConfig.N_VARS.k: n_vars,
        CegisConfig.CERTIFICATE.k: CertificateType.BARRIER,
        CegisConfig.TIME_DOMAIN.k: TimeDomain.CONTINUOUS,
        CegisConfig.VERIFIER.k: VerifierType.DREAL,
        CegisConfig.ACTIVATION.k: activations,
        CegisConfig.SYSTEM.k: F,
        CegisConfig.N_HIDDEN_NEURONS.k: n_hidden_neurons,
        CegisConfig.SYMMETRIC_BELT.k: True,
    }

    start = timeit.default_timer()
    c = Cegis(**opts)
    c.solve()
    stop = timeit.default_timer()
    print("Elapsed Time: {}".format(stop - start))

    plot_benchmark(
        system,
        D,
        certificate=c.learner,
        xrange=[0, 1.5],
        yrange=[0, 1.5],
        levels=[0],
    )


if __name__ == "__main__":
    torch.manual_seed(167)
    test_lnn()
