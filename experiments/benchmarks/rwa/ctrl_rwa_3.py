# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
from experiments.benchmarks import models
from fossil import domains
from fossil import certificate
from fossil import main, control
from fossil.consts import *


def test_lnn(args):
    ###########################################
    ###
    #############################################
    ol_system = models.ThirdOrder
    n_vars = ol_system.n_vars
    system = control.GeneralClosedLoopModel.prepare_from_open(ol_system())

    XD = domains.Rectangle([-6, -6, -6], [6, 6, 6])
    XS = domains.Rectangle([-5, -5, -5], [5, 5, 5])
    XI = domains.Rectangle([-1.2, -1.2, -1.2], [1.2, 1.2, 1.2])
    XG = domains.Rectangle([-0.3, -0.3, -0.3], [0.3, 0.3, 0.3])

    SU = domains.SetMinus(XD, XS)  # Data for unsafe set
    SD = domains.SetMinus(XS, XG)  # Data for lie set

    # define NN parameters
    activations = [ActivationType.POLY_8]
    n_hidden_neurons = [15] * len(activations)

    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XS_BORDER: XS,
        certificate.XS: XS,
        certificate.XG: XG,
    }
    data = {
        certificate.XD: XD._generate_data(1000),
        certificate.XI: XI._generate_data(100),
        certificate.XU: SU._generate_data(1000),
    }

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [5] * len(activations)

    opts = CegisConfig(
        DOMAINS=sets,
        DATA=data,
        SYSTEM=system,
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.RWS,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        CEGIS_MAX_ITERS=25,
        CTRLAYER=[8, 1],
        CTRLACTIVATION=[ActivationType.LINEAR],
    )

    main.run_benchmark(
        opts,
        record=args.record,
        plot=args.plot,
        concurrent=args.concurrent,
        repeat=args.repeat,
    )


if __name__ == "__main__":
    args = main.parse_benchmark_args()
    test_lnn(args)
