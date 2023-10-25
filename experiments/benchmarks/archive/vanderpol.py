# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pylint: disable=not-callable

import fossil.domains as domains
from fossil import plotting
from experiments.benchmarks.models import VanDerPol
from fossil import main, control
from fossil.consts import *
from fossil import certificate


def test_lnn(args):
    ###########################################
    ### Converges in 0.1s in second step
    ### To be improved. Just to demonstrate/ experiment with plotting functionality
    ###
    #############################################
    n_vars = 2

    system = VanDerPol
    batch_size = 200

    # XU = sets.SetMinus(sets.Rectangle([0, 0], [1.2, 1.2]), sets.Sphere([0.6, 0.6], 0.4))

    XD = domains.Rectangle([-3, -3], [3, 3])
    XU = domains.Torus([0, 0], 3, 2.5)
    # XU = domains.SetMinus(XD, domains.Rectangle([-2.5, -2.5], [2.5, 2.5]))
    XI = domains.Sphere([0.1, 0.1], 0.5)

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
    activations = [ActivationType.SIGMOID, ActivationType.SQUARE]
    n_hidden_neurons = [6] * len(activations)
    opts = CegisConfig(
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
        N_VARS=system.n_vars,
        CERTIFICATE=CertificateType.BARRIER,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        SYMMETRIC_BELT=False,
        CEGIS_MAX_ITERS=10,
    )

    main.run_benchmark(
        opts,
        record=args.record,
        plot=True,
        concurrent=args.concurrent,
        repeat=args.repeat,
    )


if __name__ == "__main__":
    args = main.parse_benchmark_args()
    test_lnn(args)
