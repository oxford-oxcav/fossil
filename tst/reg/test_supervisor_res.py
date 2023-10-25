# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import timeit

# pylint: disable=not-callable
import torch

import fossil.plotting as plotting
from experiments.benchmarks.benchmark_ctrl import trivial_ctrllyap
from experiments.benchmarks.benchmarks_lyap import *
from fossil.cegis_supervisor import CegisSupervisorQ
from fossil.consts import *


def test_lnn():
    benchmark = nonpoly0_lyap
    n_vars = 2
    system = benchmark
    f = system

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [2] * len(activations)

    start = timeit.default_timer()
    opts = CegisConfig(
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.LYAPUNOV,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        SYSTEM=system,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        LLO=True,
        CEGIS_MAX_ITERS=1,
    )
    sup = CegisSupervisorQ(max_P=1)
    res = sup.solve(opts)
    stop = timeit.default_timer()
    print("Elapsed Time: {}".format(stop - start))
    print(res)


def test_lnn_ctrl():
    benchmark = trivial_ctrllyap
    n_vars = 2
    system = benchmark
    f = system

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [2] * len(activations)

    start = timeit.default_timer()
    opts = CegisConfig(
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.LYAPUNOV,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        SYSTEM=system,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        LLO=True,
        CEGIS_MAX_ITERS=1,
        CTRLAYER=[15, 2],
        CTRLACTIVATION=[ActivationType.LINEAR],
    )
    sup = CegisSupervisorQ(max_P=4)
    res = sup.solve(opts)
    stop = timeit.default_timer()
    print("Elapsed Time: {}".format(stop - start))
    print(res)


if __name__ == "__main__":
    torch.manual_seed(167)
    torch.set_num_threads(1)
    test_lnn()
    test_lnn_ctrl()
