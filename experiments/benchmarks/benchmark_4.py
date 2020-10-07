import torch
import timeit
from src.lyap.cegis_lyap import Cegis
from experiments.benchmarks.benchmarks_lyap import *
from src.shared.activations import ActivationType
from src.shared.consts import VerifierType, LearnerType
from src.shared.cegis_values import CegisConfig
from src.plots.plot_lyap import plot_lyce
from functools import partial


def test_lnn():
    batch_size = 500
    benchmark = benchmark_4
    n_vars = 2
    system = partial(benchmark, batch_size)

    # define domain constraints
    outer_radius = 10
    inner_radius = 0.01

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [5] * len(activations)

    learner_type = LearnerType.NN
    verifier_type = VerifierType.DREAL
    opts = {CegisConfig.SP_HANDLE.k: False, CegisConfig.LLO.k: True}

    start = timeit.default_timer()
    c = Cegis(n_vars, system, learner_type, activations, n_hidden_neurons,
              verifier_type, inner_radius, outer_radius,
              **opts)
    state, vars, f_learner, iters = c.solve()
    stop = timeit.default_timer()

    # plotting -- only for 2-d systems
    if len(vars) == 2 and state['found']:
        plot_lyce(np.array(vars), state['V'],
                      state['V_dot'], f_learner)

    print('Elapsed Time: {}'.format(stop-start))


if __name__ == '__main__':
    torch.manual_seed(167)
    test_lnn()
