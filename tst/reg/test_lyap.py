import torch
import timeit
from src.lyap.cegis_lyap import Cegis
from experiments.benchmarks.benchmarks_lyap import *
from src.shared.activations import ActivationType
from src.shared.consts import VerifierType, LearnerType
from functools import partial


def test_lnn(benchmark, n_vars):
    batch_size = 500

    system = partial(benchmark, batch_size)

    # define domain constraints
    outer_radius = 10
    inner_radius = 0.1

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [10] * len(activations)

    learner_type = LearnerType.NN
    verifier_type = VerifierType.Z3

    factors = None

    start = timeit.default_timer()
    c = Cegis(n_vars, system, learner_type, activations, n_hidden_neurons,
              verifier_type, inner_radius, outer_radius,
              factors=factors, sp_handle=True)
    c.solve()
    stop = timeit.default_timer()
    print('Elapsed Time: {}'.format(stop-start))


if __name__ == '__main__':
    torch.manual_seed(167)
    # test_lnn(benchmark=nonpoly0, n_vars=2)
    # test_lnn(benchmark=nonpoly1, n_vars=2)
    # test_lnn(benchmark=nonpoly2, n_vars=3)
    test_lnn(benchmark=nonpoly3, n_vars=3)
