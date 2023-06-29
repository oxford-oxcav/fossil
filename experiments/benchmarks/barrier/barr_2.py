# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable


import torch
from experiments.benchmarks import models
from src import domains
from src import certificate
from src import main
from src.consts import *


def test_lnn():
    batch_size = 500

    system = models.Barr2

    class Barr2Domain(domains.Set):
        def generate_domain(self, v):
            x, y = v
            f = self.set_functions(v)
            return f["And"](-2 <= x, y <= 2)

        def generate_data(self, batch_size):
            x_comp = -2 + torch.randn(batch_size, 1) ** 2
            y_comp = 2 - torch.randn(batch_size, 1) ** 2
            dom = torch.cat([x_comp, y_comp], dim=1)
            return dom

    XD = Barr2Domain()
    XI = domains.Sphere([-0.5, 0.5], 0.4)
    XU = domains.Sphere([0.7, -0.7], 0.3)

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
    # < 5 seconds, iteration 13
    ###############################

    activations = [ActivationType.TANH]
    hidden_neurons = [15]
    opts = CegisConfig(
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
        N_VARS=system.n_vars,
        CERTIFICATE=CertificateType.BARRIER,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=hidden_neurons,
        SYMMETRIC_BELT=True,
        CEGIS_MAX_ITERS=25,
    )

    main.run_benchmark(opts, record=True, plot=True, repeat=1, concurrent=True)


if __name__ == "__main__":
    test_lnn()
