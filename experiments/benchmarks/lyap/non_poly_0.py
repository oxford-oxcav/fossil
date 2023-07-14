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
    system = models.NonPoly0
    X = domains.Torus([0, 0], 1, 0.01)
    domain = {certificate.XD: X}
    data = {certificate.XD: X._generate_data(1000)}

    # define NN parameters
    activations = [ActivationType.EVEN_POLY_6]
    n_hidden_neurons = [6] * len(activations)

    ###
    #
    ###
    opts = CegisConfig(
        SYSTEM=system,
        DOMAINS=domain,
        DATA=data,
        N_VARS=system.n_vars,
        CERTIFICATE=CertificateType.LYAPUNOV,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        LLO=True,
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
