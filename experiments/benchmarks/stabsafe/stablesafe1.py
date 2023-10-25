# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pylint: disable=not-callable

from experiments.benchmarks import models
from fossil import main, control
import fossil.domains as domains
import fossil.certificate as certificate
from fossil.consts import *


def test_lnn(args):
    n_vars = 3

    system = models.NonPoly3
    batch_size = 500

    XD = domains.Torus([0, 0, 0], 3, 0.01)

    # XU = domains.SetMinus(domains.Rectangle([0, 0], [1.2, 1.2]), domains.Sphere([0.6, 0.6], 0.4))
    XU = domains.Union(
        domains.Sphere([0.4, 0.4, 0.4], 0.2), domains.Sphere([-0.4, 0.4, 0.4], 0.2)
    )
    XI = domains.Sphere([-0.9, -0.9, -0.9], 1.0)

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
    activations = [ActivationType.SQUARE]
    activations_alt = [ActivationType.TANH]
    n_hidden_neurons = [6] * len(activations)
    n_hidden_neurons_alt = [5] * len(activations_alt)

    opts = CegisConfig(
        N_VARS=n_vars,
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
        SYMMETRIC_BELT=False,
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
