# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
import torch
import timeit
from time import sleep
from src.shared.components.cegis import Cegis
from src.shared.CegisSupervisor import CegisSupervisorQ, CegisSupervisor
from experiments.benchmarks.benchmarks_lyap import *
from src.shared.consts import *
import src.plots.plot_fcns as plotting


def test_lnn():
    benchmark = nonpoly0_lyap
    n_vars = 2
    system = benchmark
    f = system()[0]

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
    sup = CegisSupervisorQ(max_P=4)
    res = sup.run(opts)
    print(res)
    stop = timeit.default_timer()
    print("Elapsed Time: {}".format(stop - start))

    i = res["id"]
    for key in res.keys():
        assert str(i) in key or "id" in key

    cert = res["cert" + str(i)]

    plotting.benchmark(
        f,
        cert,
        {},
        levels=[0.1, 0.5, 1],
        xrange=[-3, 3],
        yrange=[-3, 3],
    )


if __name__ == "__main__":
    # torch.manual_seed(167)
    torch.set_num_threads(1)
    test_lnn()
