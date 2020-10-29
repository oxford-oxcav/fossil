from experiments.benchmarks.benchmarks_lyap import six_poly
from src.shared.consts import VerifierType, LearnerType
from src.shared.activations import ActivationType
from src.shared.cegis_values import CegisConfig, CegisStateKeys
from src.lyap.cegis_lyap import Cegis
from functools import partial
import timeit
import torch


def test_lnn():

    batch_size = 1000
    benchmark = six_poly
    n_vars = 6
    system = partial(benchmark, batch_size)

    # define domain constraints
    outer_radius = 10
    inner_radius = 0.0

    # define NN parameters
    activations = [ActivationType.LINEAR, ActivationType.LINEAR, ActivationType.SQUARE]
    n_hidden_neurons = [20] * len(activations)

    opts = {
        CegisConfig.N_VARS.k: n_vars,
        CegisConfig.LEARNER.k: LearnerType.NN,
        CegisConfig.VERIFIER.k: VerifierType.DREAL,
        CegisConfig.ACTIVATION.k: activations,
        CegisConfig.SYSTEM.k: system,
        CegisConfig.N_HIDDEN_NEURONS.k: n_hidden_neurons,
        CegisConfig.SP_HANDLE.k: False,
        CegisConfig.INNER_RADIUS.k: inner_radius,
        CegisConfig.OUTER_RADIUS.k: outer_radius,
        CegisConfig.LLO.k: True,
        CegisConfig.ROUNDING.k: -1,
    }

    start = timeit.default_timer()
    c = Cegis(**opts)
    c.solve()
    stop = timeit.default_timer()
    print('Elapsed Time: {}'.format(stop-start))


if __name__ == '__main__':
    torch.manual_seed(167)
    test_lnn()
