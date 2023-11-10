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
    n_vars = 2

    ol_system = models.SineModel
    system = control.GeneralClosedLoopModel.prepare_from_open(ol_system())

    batch_size = 500

    XD = domains.Rectangle([-3, -3], [3, 3])
    XS = domains.Rectangle([-2.5, -2.5], [2.5, 2.5])
    XI = domains.Rectangle([-0.6, -0.6], [0.6, 0.6])
    XG = domains.Rectangle([-0.01, -0.01], [0.01, 0.01])

    SU = domains.SetMinus(XD, XS)  # Data for unsafe set
    SD = domains.SetMinus(XS, XG)  # Data for lie set
    sets = {
        "lie": XD,
        "init": XI,
        "safe_border": XS,
        "safe": XS,
        "goal": XG,
    }
    data = {
        "lie": SD._generate_data(batch_size),
        "init": XI._generate_data(1000),
        "unsafe": SU._generate_data(1000),
    }

    # define NN parameters
    activations = [ActivationType.SIGMOID]
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
        CTRLAYER=[8, 2],
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
