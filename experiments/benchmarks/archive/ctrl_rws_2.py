# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import timeit

# pylint: disable=not-callable
import torch

from src import domains
from src import certificate
from src import main
from experiments.benchmarks import models
from experiments.benchmarks.models import SecondOrder
from src.consts import *


def test_lnn(args):
    ###########################################
    ###
    #############################################
    n_vars = 2

    ol_system = SecondOrder()
    system = lambda ctrl: models.GeneralClosedLoopModel(ol_system, ctrl)

    XD = domains.Torus([0, 0], 2, 0.05)
    XS = domains.Rectangle([-1, -1], [1, 1])
    XI = domains.Rectangle([-0.5, -0.5], [0.5, 0.5])
    XG = domains.Rectangle([-0.05, -0.05], [0.05, 0.05])

    SU = domains.SetMinus(XD, XS)  # Data for unsafe set
    SD = domains.SetMinus(XS, XG)  # Data for lie set

    sets = {
        certificate.XD: XD,
        # certificate.XI: XI,
        # certificate.XS_BORDER: XS,
        # certificate.XS: XS,
        # certificate.XG: XG,
    }
    data = {
        certificate.XD: XD._generate_data(500),
        # certificate.XI: XI._generate_data(100),
        # certificate.XU: SU._generate_data(1000),
    }
    F = lambda *args: (system(*args), set, data, sets.inf_bounds_n(2))

    # define NN parameters
    activations = [ActivationType.EVEN_POLY_4]
    n_hidden_neurons = [8] * len(activations)

    opts = CegisConfig(
        DOMAINS=sets,
        DATA=data,
        SYSTEM=system,
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.LYAPUNOV,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        CEGIS_MAX_ITERS=10,
        CTRLAYER=[8, 1],
        CTRLACTIVATION=[ActivationType.LINEAR],
        LLO=True,
    )

    main.run_benchmark(
        opts,
        record=args.record,
        plot=True,
        concurrent=args.concurrent,
        repeat=args.repeat,
        xrange=[-1, 1],
        yrange=[-1, 1],
    )


if __name__ == "__main__":
    for i in range(5):
        args = main.parse_benchmark_args()
        test_lnn(args)
