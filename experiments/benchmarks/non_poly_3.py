import timeit
from src.lyap.cegis_lyap import Cegis
from experiments.benchmarks.benchmarks_lyap import *
from src.shared.activations import ActivationType
from src.shared.consts import VerifierType, LearnerType
from functools import partial
from src.shared.cegis_values import CegisConfig


def test_lnn():
    batch_size = 500
    benchmark = nonpoly3
    n_vars = 3
    system = partial(benchmark, batch_size)

    # define domain constraints
    outer_radius = 10
    inner_radius = 0.01

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [3] * len(activations)

    learner_type = LearnerType.NN
    verifier_type = VerifierType.Z3
    opts = {CegisConfig.SP_HANDLE.k: True, CegisConfig.LLO.k: True}

    start = timeit.default_timer()
    c = Cegis(n_vars, system, learner_type, activations, n_hidden_neurons,
              verifier_type, inner_radius, outer_radius,
              **opts)
    c.solve()
    stop = timeit.default_timer()
    print('Elapsed Time: {}'.format(stop-start))


if __name__ == '__main__':
    torch.manual_seed(167)
    test_lnn()
