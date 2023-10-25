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
    ol_system = models.ThirdOrder
    system = control.GeneralClosedLoopModel.prepare_from_open(ol_system())
    XD = domains.Rectangle([-6, -6, -6], [6, 6, 6])
    XS = domains.Rectangle([-5, -5, -5], [5, 5, 5])
    XU = domains.Complement(XS)
    XI = domains.Rectangle([-1.2, -1.2, -1.2], [1.2, 1.2, 1.2])
    XR = XI

    SU = domains.SetMinus(XD, XS)  # Data for unsafe set

    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XU: XU,
    }
    data = {
        certificate.XD: XD._generate_data(3000),
        certificate.XR: XI._generate_data(3000),
        certificate.XI: XI._generate_data(3000),
        certificate.XU: SU._generate_data(3000),
    }

    # define NN parameters
    activations = [ActivationType.SQUARE]
    activations_alt = [ActivationType.TANH]
    n_hidden_neurons = [10] * len(activations)
    n_hidden_neurons_alt = [8] * len(activations_alt)

    opts = CegisConfig(
        N_VARS=3,
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
        CTRLAYER=[8, 1],
        CTRLACTIVATION=[ActivationType.LINEAR],
        SYMMETRIC_BELT=False,
        CEGIS_MAX_ITERS=100,
        LLO=True,
        VERBOSE=False,
    )

    main.run_benchmark(
        opts,
        record=args.record,
        plot=args.record,
        concurrent=args.concurrent,
        repeat=args.repeat,
    )


if __name__ == "__main__":
    args = main.parse_benchmark_args()
    test_lnn(args)
