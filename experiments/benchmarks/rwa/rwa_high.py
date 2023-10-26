# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
from experiments.benchmarks import models
from src import domains
from src import certificate
from src import main
from src.consts import *


def test_lnn(args):
    system = models.Satellite
    n_vars = system.n_vars
    batch_size = 5000

    XD = domains.Rectangle([-2.5] * n_vars, [2.5] * n_vars)
    XI = domains.Sphere([0] * n_vars, 0.3)
    XU = domains.OpenSphere([-2] * n_vars, 0.1)
    XS = domains.Rectangle([-2] * n_vars, [2] * n_vars)

    XG = domains.Sphere([0] * n_vars, 0.1)
    SU = domains.Union(XD, XU)

    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XS_BORDER: XS,
        certificate.XS: XS,
        certificate.XG: XG,
    }
    data = {
        certificate.XD: XD._generate_data(batch_size),
        certificate.XI: XI._sample_border(batch_size),
        certificate.XU: XS._sample_border(batch_size),
    }
    # SU = domains.SetMinus(XD, XS)  # Data for unsafe set

    # sets = {
    #     certificate.XD: XD,
    #     certificate.XI: XI,
    #     certificate.XS_BORDER: XS,
    #     certificate.XS: XS,
    #     certificate.XG: XG,
    # }
    # data = {
    #     certificate.XD: XD._generate_data(2000),
    #     certificate.XI: XI._sample_border(2000),
    #     certificate.XU: XU._sample_border(2000),
    # }

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [10] * len(activations)

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
        CEGIS_MAX_ITERS=50,
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
