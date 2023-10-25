# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable

import torch

from experiments.benchmarks import models
from fossil import domains
from fossil import certificate
from fossil import main, control
from fossil.consts import *


class Domain(domains.Set):
    dimension = 3

    def generate_domain(self, v):
        x, y, phi = v
        f = self.set_functions(v)
        return f["And"](-2 <= x, x <= 2, -2 <= y, y <= 2, -1.57 <= phi, phi <= 1.57)

    def generate_data(self, batch_size):
        k = 4
        x_comp = -2 + torch.sum(torch.randn(batch_size, k) ** 2, dim=1).reshape(
            batch_size, 1
        )
        y_comp = 2 - torch.sum(torch.randn(batch_size, k) ** 2, dim=1).reshape(
            batch_size, 1
        )
        phi_comp = domains.segment([-1.57, 1.57], batch_size)
        dom = torch.cat([x_comp, y_comp, phi_comp], dim=1)
        return dom


class Init(domains.Set):
    dimension = 3

    def generate_domain(self, v):
        x, y, phi = v
        f = self.set_functions(v)
        return f["And"](
            -0.1 <= x, x <= 0.1, -2 <= y, y <= -1.8, -0.52 <= phi, phi <= 0.52
        )

    def generate_data(self, batch_size):
        x = domains.segment([-0.1, 0.1], batch_size)
        y = domains.segment([-2.0, -1.8], batch_size)
        phi = domains.segment([-0.52, 0.52], batch_size)
        return torch.cat([x, y, phi], dim=1)


class UnsafeDomain(domains.Set):
    dimension = 3

    def generate_domain(self, v):
        x, y, _phi = v
        return x**2 + y**2 <= 0.04

    def generate_data(self, batch_size):
        xy = domains.circle_init_data((0.0, 0.0), 0.04, batch_size)
        phi = domains.segment([-0.52, 0.52], batch_size)
        return torch.cat([xy, phi], dim=1)


def test_lnn(args):
    XD = Domain()
    XI = Init()
    XU = UnsafeDomain()
    batch_size = 2000
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

    ###
    #
    ###
    system = models.ObstacleAvoidance
    activations = [ActivationType.POLY_4]
    hidden_neurons = [25]
    opts = CegisConfig(
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
        N_VARS=system.n_vars,
        CERTIFICATE=CertificateType.BARRIER,
        ACTIVATION=activations,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        N_HIDDEN_NEURONS=hidden_neurons,
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
