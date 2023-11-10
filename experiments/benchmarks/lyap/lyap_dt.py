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
    open_loop = models.DTAhmadi2

    n_vars = open_loop.n_vars
    outer = 1.0
    inner = 0.1
    batch_size = 5000
    XD = domains.Torus([0.0, 0.0], outer, inner)

    sets = {
        certificate.XD: XD,
    }
    data = {
        certificate.XD: XD._generate_data(batch_size),
    }

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [2] * len(activations)

    opts = CegisConfig(
        SYSTEM=open_loop,
        DOMAINS=sets,
        DATA=data,
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.LYAPUNOV,
        LLO=True,
        TIME_DOMAIN=TimeDomain.DISCRETE,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
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
