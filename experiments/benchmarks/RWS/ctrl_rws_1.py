# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
import torch
import timeit


from src.shared.components.cegis import Cegis
from experiments.benchmarks.models import Linear1
import experiments.benchmarks.domain_fcns as sets
from experiments.benchmarks import models
from src.shared.consts import *
import src.plots.plot_fcns as plotting


def test_lnn():
    ###########################################
    ###
    #############################################
    n_vars = 2

    ol_system = Linear1()
    system = lambda ctrl: models.GeneralClosedLoopModel(ol_system, ctrl)

    XD = sets.Rectangle([-1.5, -1.5], [1.5, 1.5])
    XS = sets.Rectangle([-1, -1], [1, 1])
    XI = sets.Rectangle([-0.5, -0.5], [0.5, 0.5])
    XG = sets.Rectangle([-0.1, -0.1], [0.1, 0.1])

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
        "lie": XD.generate_data(1000),
        "init": XI.generate_data(1000),
        "unsafe": SU.generate_data(1000),
    }
    F = lambda *args: (system(*args), symbolic_domains, data, sets.inf_bounds_n(2))

    # define NN parameters
    activations = [ActivationType.LIN_TO_QUARTIC]
    n_hidden_neurons = [18] * len(activations)

    opts = CegisConfig(
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.RWS,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        SYSTEM=F,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        CEGIS_MAX_ITERS=10,
        CTRLAYER=[8, 1],
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
        xrange=[-1.1, 1.1],
        yrange=[-1.1, 1.1],
        levels=[0],
    )

    f_sym = c.f.to_sympy()
    print(f_sym)


if __name__ == "__main__":
    torch.manual_seed(167)
    test_lnn()
