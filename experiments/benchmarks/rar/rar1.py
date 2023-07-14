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
    n_vars = 2

    system = models.Linear1LQR
    batch_size = 500

    XD = domains.Rectangle([-1.5, -1.5], [1.5, 1.5])
    XS = domains.Rectangle([-1, -1], [1, 1])
    XI = domains.Rectangle([-0.5, -0.5], [0.5, 0.5])
    XG = domains.Rectangle([-0.1, -0.1], [0.1, 0.1])
    XF = domains.Rectangle([-0.4, -0.4], [0.4, 0.4])

    SU = domains.SetMinus(XD, XS)  # Data for unsafe set
    SD = domains.SetMinus(XS, XG)  # Data for lie set
    SU2 = domains.SetMinus(XD, XF)
    from matplotlib import pyplot as plt

    # S = SU2.generate_data(1000)
    # plt.plot(S[:, 0], S[:, 1], "x")
    # plt.show()

    sets = {
        "lie": XD,
        "init": XI,
        "safe_border": XS,
        "safe": XS,
        "goal": XG,
        "final": XF,
    }
    data = {
        "lie": SD._generate_data(batch_size),
        "init": XI._generate_data(500),
        "unsafe": SU._generate_data(1000),
        "goal": XG._generate_data(300),
        "final": SU2._generate_data(100),
    }

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [4] * len(activations)

    activations_alt = [ActivationType.POLY_4]
    n_hidden_neurons_alt = [4] * len(activations_alt)

    opts = CegisConfig(
        DOMAINS=sets,
        DATA=data,
        SYSTEM=system,
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.RAR,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        ACTIVATION_ALT=activations_alt,
        N_HIDDEN_NEURONS_ALT=n_hidden_neurons_alt,
        CEGIS_MAX_ITERS=100,
        SYMMETRIC_BELT=False,
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
