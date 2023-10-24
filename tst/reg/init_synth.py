# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import traceback
import timeit
from fossil.lyap.cegis_lyap import Cegis as Cegis_for_lyap
from fossil.barrier.cegis_barrier import Cegis as Cegis_for_bc
from fossil.consts import *
from functools import partial


def lyap_synthesis(benchmark, n_vars):
    batch_size = 500
    system = partial(benchmark, batch_size=batch_size)

    # compute equilibria
    # f, _, __ = system(functions=SympySolver.solver_fncts())
    # f_sp = partial(f, SympySolver.solver_fncts())
    # x_sp = [sp.Symbol('x%d' % i) for i in range(n_vars)]
    # equilibria = compute_equilibria(f_sp(np.array(x_sp)), x_sp)
    # real_eq = check_real_solutions(equilibria, x_sp)
    # real_eq = dict_to_array(real_eq, n_vars)

    # define domain constraints
    outer_radius = 10
    inner_radius = 0.1

    # define NN parameters
    activations = [ActivationType.LINEAR]
    n_hidden_neurons = [5] * len(activations)

    learner_type = LearnerType.CONTINUOUS
    verifier_type = VerifierType.Z3

    """
    with factors = 'quadratic'
    the candidate Lyap is (x-eq0)^2 * ... * (x-eqN)^2 * NN(x)
    by passing factors = 'linear'
    have (x-eq0) * ... * (x-eqN) * NN(x)
    """
    factors = None

    start = timeit.default_timer()
    c = Cegis_for_lyap(
        n_vars,
        system,
        learner_type,
        activations,
        n_hidden_neurons,
        verifier_type,
        inner_radius,
        outer_radius,
        factors=factors,
        eq=None,
    )
    c.solve()
    stop = timeit.default_timer()
    print("Elapsed Time: {}".format(stop - start))


def barrier_synthesis(benchmark, n_vars):
    MIN_TO_SEC = 60
    batch_size = 500
    system = partial(benchmark, batch_size)
    activations = [
        ActivationType.LINEAR,
        ActivationType.POLY_3,
        ActivationType.LINEAR,
    ]
    hidden_neurons = [2] * len(activations)
    try:
        start = timeit.default_timer()
        opts = CegisConfig(
            N_VARS=n_vars,
            TIME_DOMAIN=TimeDomain.CONTINUOUS,
            VERIFIER=VerifierType.Z3,
            ACTIVATION=activations,
            SYSTEM=system,
            N_HIDDEN_NEURONS=hidden_neurons,
            CEGIS_MAX_TIME_S=30 * MIN_TO_SEC,
        )
        c = Cegis_for_bc(**opts)
        _, found, _ = c.solve()
        end = timeit.default_timer()

        print("Elapsed Time: {}".format(end - start))
        print("Found? {}".format(found))
    except Exception as _:
        print(traceback.format_exc())
