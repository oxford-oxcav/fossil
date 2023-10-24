# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import timeit

# pylint: disable=not-callable
import torch

from fossil import domains
from fossil import certificate
from fossil import main, control
from experiments.benchmarks import models
from experiments.benchmarks.models import SecondOrder
from fossil.consts import *


class RWA4Unsafe(domains.Sphere):
    def __init__(self):
        self.centre = [0.5, 0.5]
        self.radius = 0.2
        self.dimension = len(self.centre)

    def generate_domain(self, x):
        f = self.set_functions(x)
        return f["And"](
            sum([(x[i] - self.centre[i]) ** 2 for i in range(len(x))])
            < self.radius**2
        )

    def generate_boundary(self, x):
        f = self.set_functions(x)
        return f["And"](
            sum([(x[i] - self.centre[i]) ** 2 for i in range(len(x))])
            == self.radius**2
        )

    def generate_data(self, batch_size):
        return domains.round_init_data(self.centre, self.radius**2, batch_size)


def test_lnn(args):
    ###########################################
    ###
    #############################################
    n_vars = 2

    ol_system = SecondOrder
    system = control.GeneralClosedLoopModel.prepare_from_open(ol_system())

    XD = domains.Rectangle([-1.5, -1.5], [1.5, 1.5])
    XI = domains.Rectangle([-0.5, -0.5], [-0.1, -0.1])
    XG = domains.Rectangle([-0.05, -0.05], [0.05, 0.05])
    XU = RWA4Unsafe()
    XS = domains.SetMinus(XD, XU)  # Safe set

    SD = domains.SetMinus(XS, XG)  # Data for lie set

    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XS_BORDER: XS,
        certificate.XS: XS,
        certificate.XG: XG,
    }
    data = {
        certificate.XD: XD._generate_data(1000),
        certificate.XI: XI._generate_data(1000),
        certificate.XU: XU._generate_data(1000),
    }
    # define NN parameters
    activations = [ActivationType.SIGMOID, ActivationType.SQUARE]
    n_hidden_neurons = [4] * len(activations)

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
        CTRLAYER=[8, 1],
        CTRLACTIVATION=[ActivationType.LINEAR],
    )

    main.run_benchmark(
        opts,
        record=args.record,
        plot=args.plot,
        concurrent=args.concurrent,
        repeat=args.repeat,
        xrange=[-2, 2],
        yrange=[-2, 2],
    )


if __name__ == "__main__":
    args = main.parse_benchmark_args()
    test_lnn(args)
