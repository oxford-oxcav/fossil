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
from experiments.benchmarks import models
from experiments.benchmarks.models import ThirdOrder
from src.cegis import Cegis
from src.consts import *


def test_lnn():
    ###########################################
    ###
    #############################################
    ol_system = ThirdOrder()
    n_vars = ol_system.n_vars
    system = lambda ctrl: models.GeneralClosedLoopModel(ol_system, ctrl)

    XD = sets.Rectangle([-6, -6, -6], [6, 6, 6])
    XS = sets.Rectangle([-5, -5, -5], [5, 5, 5])
    XI = sets.Rectangle([-1.2, -1.2, -1.2], [1.2, 1.2, 1.2])
    XG = sets.Rectangle([-0.3, -0.3, -0.3], [0.3, 0.3, 0.3])

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
        "lie": SD.generate_data(1000),
        "init": XI.generate_data(1000),
        "unsafe": SU.generate_data(1000),
    }
    F = lambda *args: (system(*args), symbolic_domains, data, sets.inf_bounds_n(n_vars))

    # define NN parameters
    activations = [ActivationType.LIN_TO_OCTIC]
    n_hidden_neurons = [15] * len(activations)

    opts = CegisConfig(
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.RWS,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        SYSTEM=F,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        CEGIS_MAX_ITERS=10,
        CTRLAYER=[20, 1],
        CTRLACTIVATION=[ActivationType.SQUARE],
    )

    start = timeit.default_timer()
    c = Cegis(opts)
    c.solve()
    stop = timeit.default_timer()
    print("Elapsed Time: {}".format(stop - start))
    f_sym = c.f.to_sympy()
    print(f_sym)


if __name__ == "__main__":
    torch.manual_seed(167)
    test_lnn()
