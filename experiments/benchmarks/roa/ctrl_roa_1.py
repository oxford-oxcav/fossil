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
    ###########################################
    #
    #############################################
    n_vars = 3

    ol_system = models.LorenzSystem
    system = control.GeneralClosedLoopModel.prepare_from_open(ol_system())
    batch_size = 1000

    XD = domains.Sphere([0, 0, 0], 2)
    XI = domains.Sphere([0, 0, 0], 0.3)

    sets = {
        # certificate.XD: XD,
        certificate.XI: XI,
    }
    data = {
        certificate.XD: XD._generate_data(batch_size),
        certificate.XI: XI._sample_border(batch_size),
    }

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [8] * len(activations)

    opts = CegisConfig(
        N_VARS=n_vars,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
        CERTIFICATE=CertificateType.ROA,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        CTRLACTIVATION=[ActivationType.LINEAR],
        CTRLAYER=[8, 3],
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
