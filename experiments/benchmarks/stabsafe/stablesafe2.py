# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable

from fossil import domains
from fossil import certificate
from fossil import main, control
from experiments.benchmarks import models
from fossil.consts import *


def test_lnn(args):
    system = models.Barr3
    XD = domains.Rectangle([-3, -2], [2.5, 1])
    XR = domains.Torus([0, 0], 0.4, 0.01)
    XI = domains.Rectangle([0.4, 0.1], [0.8, 0.5])
    # XI = domains.Sphere((0, 0), 0.2)
    XU = domains.Sphere([-1, -1], 0.4)

    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XU: XU,
    }
    data = {
        certificate.XD: XD._generate_data(500),
        certificate.XR: XR._generate_data(500),
        certificate.XI: XI._generate_data(500),
        certificate.XU: XU._generate_data(500),
    }

    # define NN parameters
    activations = [ActivationType.SQUARE]
    activations_alt = [ActivationType.SIGMOID, ActivationType.SQUARE]
    n_hidden_neurons = [5] * len(activations)
    n_hidden_neurons_alt = [5] * len(activations_alt)

    opts = CegisConfig(
        N_VARS=2,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
        CERTIFICATE=CertificateType.STABLESAFE,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        ACTIVATION_ALT=activations_alt,
        N_HIDDEN_NEURONS_ALT=n_hidden_neurons_alt,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        SYMMETRIC_BELT=True,
        CEGIS_MAX_ITERS=25,
        LLO=True,
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
