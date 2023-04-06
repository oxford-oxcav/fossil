# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
from experiments.benchmarks.benchmarks_bc import barr_1
from src.shared.components.cegis import Cegis


from src.shared.consts import *

from src.shared.components.cegis import Cegis
from src.plots.plot_barriers import plot_darboux_bench
import numpy as np
import timeit
import torch


def main():
    system = barr_1
    activations = [ActivationType.LINEAR]
    hidden_neurons = [10] * len(activations)
    start = timeit.default_timer()
    opts = CegisConfig(
        N_VARS=2,
        CERTIFICATE=CertificateType.BARRIER,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        SYSTEM=system,
        N_HIDDEN_NEURONS=hidden_neurons,
        SYMMETRIC_BELT=True,
    )
    c = Cegis(opts)
    state, vars, f_learner, iters = c.solve()
    end = timeit.default_timer()
    print("Elapsed Time: {}".format(end - start))
    print("Found? {}".format(state[CegisStateKeys.found]))

    # plotting -- only for 2-d systems
    if state[CegisStateKeys.found]:
        plot_darboux_bench(np.array(vars), state[CegisStateKeys.V])


if __name__ == "__main__":
    torch.manual_seed(167)
    main()
