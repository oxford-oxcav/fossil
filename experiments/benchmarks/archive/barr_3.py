# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable

from src import domains
from src import certificate
from src import main
from experiments.benchmarks import models
from experiments.benchmarks.models import SecondOrder
from src.consts import *


class Init(domains.Set):
    def generate_domain(self, v):
        x, y = v
        f = self.set_functions(v)
        _Or = f["Or"]
        _And = f["And"]
        return _Or(
            _And((x - 1.5) ** 2 + y**2 <= 0.25),
            _And(x >= -1.8, x <= -1.2, y >= -0.1, y <= 0.1),
            _And(x >= -1.4, x <= -1.2, y >= -0.5, y <= 0.1),
        )

    def generate_data(self, batch_size):
        n0 = int(batch_size / 3)
        n1 = n0
        n2 = batch_size - (n0 + n1)
        return torch.cat(
            [
                domains.circle_init_data((1.5, 0.0), 0.25, n0),
                domains.square_init_data([[-1.8, -0.1], [-1.2, 0.1]], n1),
                domains.add_corners_2d([[-1.8, -0.1], [-1.2, 0.1]]),
                domains.square_init_data([[-1.4, -0.5], [-1.2, 0.1]], n2),
                domains.add_corners_2d([[-1.4, -0.5], [-1.2, 0.1]]),
            ]
        )


class Unsafe(domains.Set):
    def generate_domain(self, v):
        x, y = v
        f = self.set_functions(v)
        _Or = f["Or"]
        _And = f["And"]
        return _Or(
            (x + 1) ** 2 + (y + 1) ** 2 <= 0.16,
            _And(0.4 <= x, x <= 0.6, 0.1 <= y, y <= 0.5),
            _And(0.4 <= x, x <= 0.8, 0.1 <= y, y <= 0.3),
        )

    def generate_data(self, batch_size):
        n0 = int(batch_size / 3)
        n1 = n0
        n2 = batch_size - (n0 + n1)
        return torch.cat(
            [
                domains.circle_init_data((-1.0, -1.0), 0.16, n0),
                domains.square_init_data([[0.4, 0.1], [0.6, 0.5]], n1),
                domains.add_corners_2d([[0.4, 0.1], [0.6, 0.5]]),
                domains.square_init_data([[0.4, 0.1], [0.8, 0.3]], n2),
                domains.add_corners_2d([[0.4, 0.1], [0.8, 0.3]]),
            ]
        )


def test_lnn(args):
    system = models.Barr3
    XD = domains.Rectangle([-3, -2], [2.5, 1])
    XI = domains.Rectangle([0.4, 0.1], [0.8, 0.5])
    XU = domains.Sphere([-1, -1], 0.4)

    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XU: XU,
    }
    data = {
        certificate.XD: XD._generate_data(500),
        certificate.XI: XI._generate_data(500),
        certificate.XU: XU._generate_data(500),
    }

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [5] * len(activations)

    opts = CegisConfig(
        N_VARS=2,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
        CERTIFICATE=CertificateType.BARRIER,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        SYMMETRIC_BELT=True,
        CEGIS_MAX_ITERS=25,
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
