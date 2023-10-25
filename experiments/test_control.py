# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
import torch
import timeit
from fossil.cegis import Cegis
from experiments.benchmarks.benchmark_ctrl import trivial_ctrllyap


from fossil.consts import *
from fossil.analysis import Recorder


def test_lnn():
    # TEST for Control Lyapunov
    # pass the ctrl parameters from here (i.e. the main)
    benchmark = trivial_ctrllyap
    n_vars = 2
    system = benchmark

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [4] * len(activations)

    start = timeit.default_timer()
    opts = CegisConfig(
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.LYAPUNOV,
        LLO=True,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.Z3,
        ACTIVATION=activations,
        SYSTEM=system,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        CTRLAYER=[15, 2],
        CTRLACTIVATION=[ActivationType.LINEAR],
    )
    c = Cegis(opts)
    state, vars, f, iters = c.solve()
    stop = timeit.default_timer()
    print("Elapsed Time: {}".format(stop - start))
    rec = Recorder()
    rec.record(opts, state, iters, stop - start)

    # plotting -- only for 2-d systems
    return state["found"], stop - start


if __name__ == "__main__":
    torch.manual_seed(169)
    torch.set_num_threads(1)
    success = 0
    sum_T = 0
    for i in range(10):
        res, T = test_lnn()
        if res:
            success += 1
        sum_T += T
    print("Success rate: {}".format(success / 10))
    print("Average time: {}".format(sum_T / 10))
