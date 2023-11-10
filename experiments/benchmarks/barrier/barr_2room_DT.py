# Copyright (c) 2023, Alessandro Abate, Alec Edwards, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
from experiments.benchmarks import models
from fossil import domains
from fossil import certificate
from fossil import main
from fossil.consts import *


def test_lnn(args):
    batch_size = 5000
    open_loop = models.TwoRoomTemp
    n_vars = open_loop.n_vars

    XD = domains.Rectangle(lb=[18.0] * n_vars, ub=[23.0] * n_vars)
    XI = domains.Rectangle(lb=[18.0] * n_vars, ub=[19.75] * n_vars)
    XU = domains.Rectangle(lb=[22.0] * n_vars, ub=[23.0] * n_vars)

    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XU: XU,
    }
    data = {
        certificate.XD: XD._generate_data(batch_size),
        certificate.XI: XI._generate_data(batch_size),
        certificate.XU: XU._generate_data(batch_size),
    }

    # define NN parameters
    barr_activations = [ActivationType.SQUARE]
    barr_hidden_neurons = [2] * len(barr_activations)

    opts = CegisConfig(
        SYSTEM=open_loop,
        DOMAINS=sets,
        DATA=data,
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.BARRIERALT,
        TIME_DOMAIN=TimeDomain.DISCRETE,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=barr_activations,
        N_HIDDEN_NEURONS=barr_hidden_neurons,
        SYMMETRIC_BELT=False,
        CEGIS_MAX_ITERS=25,
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
