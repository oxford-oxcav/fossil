# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
import torch
import timeit
from src.shared.components.cegis import DoubleCegis
from experiments.benchmarks.benchmarks_lyap import *
from src.shared.activations import ActivationType
from src.shared.cegis_values import CegisConfig, CegisStateKeys
from src.shared.consts import VerifierType, TimeDomain, CertificateType
from src.plots.plot_lyap import plot_lyce


def test_lnn():
    benchmark_lyap = ras_demo_lyap
    benchmark_barr = ras_demo_barr
    n_vars = 2

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [10] * len(activations)

    start = timeit.default_timer()
    opts_lyap = {
        CegisConfig.N_VARS.k: n_vars,
        CegisConfig.CERTIFICATE.k: CertificateType.LYAPUNOV,
        CegisConfig.TIME_DOMAIN.k: TimeDomain.CONTINUOUS,
        CegisConfig.VERIFIER.k: VerifierType.DREAL,
        CegisConfig.ACTIVATION.k: activations,
        CegisConfig.SYSTEM.k: benchmark_lyap,
        CegisConfig.N_HIDDEN_NEURONS.k: n_hidden_neurons,
    }
    opts_barr = {
        CegisConfig.N_VARS.k: n_vars,
        CegisConfig.CERTIFICATE.k: CertificateType.BARRIER,
        CegisConfig.TIME_DOMAIN.k: TimeDomain.CONTINUOUS,
        CegisConfig.VERIFIER.k: VerifierType.DREAL,
        CegisConfig.ACTIVATION.k: activations,
        CegisConfig.SYSTEM.k: benchmark_barr,
        CegisConfig.N_HIDDEN_NEURONS.k: n_hidden_neurons,
    }

    c = DoubleCegis(opts_lyap, opts_barr)
    res = c.solve()
    stop = timeit.default_timer()
    print("Elapsed Time: {}".format(stop - start))


if __name__ == "__main__":
    torch.manual_seed(167)
    test_lnn()
