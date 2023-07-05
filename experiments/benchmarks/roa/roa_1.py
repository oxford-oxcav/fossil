# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pylint: disable=not-callable

from experiments.benchmarks import models
from src import main
import src.domains as domains
import src.certificate as certificate
from src.consts import *


def test_lnn():
    ###########################################
    ### Converges in 1.6s in second step
    ### Trivial example
    ### Currently DoubleCegis does not work with consolidator
    #############################################
    n_vars = 2

    system = models.NonPoly1
    batch_size = 500

    XD = domains.Torus([0, 0], 2, 0.01)
    XI = domains.Torus([0, 0], 0.5, 0.01)

    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
    }
    data = {
        certificate.XD: XD._generate_data(batch_size),
        certificate.XI: XI._sample_border(batch_size),
    }

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [10] * len(activations)

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
        CEGIS_MAX_ITERS=10,
        LLO=True,
    )

    main.run_benchmark(
        opts, record=False, plot=True, repeat=1, xrange=[-5, 5], yrange=[-5, 5]
    )


if __name__ == "__main__":
    test_lnn()
