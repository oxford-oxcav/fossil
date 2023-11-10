# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable

from fossil import domains
from fossil import certificate
from fossil import main
from experiments.benchmarks import models
from fossil.consts import *


def test_lnn(args):
    system = models.Barr3
    XD = domains.Rectangle([-3, -2], [2.5, 1])
    XI = domains.Union(
        domains.Sphere([1.5, 0], 0.5),
        domains.Union(
            domains.Rectangle([-1.8, -0.1], [-1.2, 0.1]),
            domains.Rectangle([-1.4, -0.5], [-1.2, 0.1]),
        ),
    )

    XU = domains.Union(
        domains.Sphere([-1, -1], 0.4),
        domains.Union(
            domains.Rectangle([0.4, 0.1], [0.6, 0.5]),
            domains.Rectangle([0.4, 0.1], [0.8, 0.3]),
        ),
    )

    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XU: XU,
    }
    data = {
        certificate.XD: XD._generate_data(1000),
        certificate.XI: XI._generate_data(400),
        certificate.XU: XU._generate_data(400),
    }

    # define NN parameters
    activations = [ActivationType.SIGMOID, ActivationType.SIGMOID]
    n_hidden_neurons = [10] * len(activations)

    opts = CegisConfig(
        N_VARS=2,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
        CERTIFICATE=CertificateType.BARRIER,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        SYMMETRIC_BELT=False,
        CEGIS_MAX_ITERS=25,
        VERBOSE=0,
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
