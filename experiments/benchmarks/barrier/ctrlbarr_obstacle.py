# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
import numpy as np

from experiments.benchmarks import models
from fossil import domains
from fossil import certificate
from fossil import main, control
from fossil.consts import *


def test_lnn(args):
    batch_size = 1000
    open_loop = models.CtrlObstacleAvoidance

    XD = domains.Rectangle(lb=[-10.0, -10.0, -np.pi], ub=[10.0, 10.0, np.pi])
    XI = domains.Rectangle(lb=[4.0, 4.0, -np.pi / 2], ub=[6.0, 6.0, np.pi / 2])
    XU = domains.Rectangle(lb=[-9, -9, -np.pi / 2], ub=[-7.0, -6.0, np.pi / 2])
    lowers = [-0.05, -0.05, -0.05]
    uppers = [0.05, 0.05, 0.05]

    system = control.GeneralClosedLoopModel.prepare_from_open(open_loop())

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

    ###############################
    #
    ###############################

    # using trajectory control
    n_vars = 3

    # define NN parameters
    barr_activations = [ActivationType.TANH]
    barr_hidden_neurons = [15] * len(barr_activations)

    # ctrl params
    n_ctrl_inputs = 1

    opts = CegisConfig(
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.BARRIER,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=barr_activations,
        N_HIDDEN_NEURONS=barr_hidden_neurons,
        CTRLAYER=[5, n_ctrl_inputs],
        CTRLACTIVATION=[ActivationType.TANH],
        SYMMETRIC_BELT=True,
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
