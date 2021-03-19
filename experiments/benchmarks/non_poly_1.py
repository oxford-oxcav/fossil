# pylint: disable=not-callable
from src.shared.consts import VerifierType, LearnerType, ConsolidatorType, TranslatorType
from experiments.benchmarks.benchmarks_lyap import *
from src.shared.activations import ActivationType
from src.shared.cegis_values import CegisConfig, CegisStateKeys
from src.plots.plot_lyap import plot_lyce
from src.lyap.cegis_lyap import Cegis
from src.lyap.utils import check_sympy_expression
from functools import partial

import numpy as np
import timeit


def test_lnn():
    batch_size = 500
    benchmark = nonpoly1
    n_vars = 2
    system = partial(benchmark, batch_size)

    # define domain constraints
    outer_radius = 10
    inner_radius = 0.01

    # define NN parameters
    activations = [ActivationType.LINEAR, ActivationType.SQUARE]
    n_hidden_neurons = [20] * len(activations)

    opts = {
        CegisConfig.N_VARS.k: n_vars,
        CegisConfig.LEARNER.k: LearnerType.NN,
        CegisConfig.VERIFIER.k: VerifierType.Z3,
        CegisConfig.CONSOLIDATOR.k: ConsolidatorType.DEFAULT,
        CegisConfig.TRANSLATOR.k: TranslatorType.DEFAULT,
        CegisConfig.ACTIVATION.k: activations,
        CegisConfig.SYSTEM.k: system,
        CegisConfig.N_HIDDEN_NEURONS.k: n_hidden_neurons,
        CegisConfig.SP_HANDLE.k: False,
        CegisConfig.INNER_RADIUS.k: inner_radius,
        CegisConfig.OUTER_RADIUS.k: outer_radius,
        CegisConfig.LLO.k: True,
    }
    start = timeit.default_timer()
    c = Cegis(**opts)
    state, vars, f_learner, iters = c.solve()
    stop = timeit.default_timer()
    print('Elapsed Time: {}'.format(stop-start))

    # plotting -- only for 2-d systems
    if len(vars) == 2 and state[CegisStateKeys.found]:
        V, Vdot = check_sympy_expression(state, system)
        plot_lyce(np.array(vars), V, Vdot, f_learner)


if __name__ == '__main__':
    torch.manual_seed(167)
    test_lnn()
