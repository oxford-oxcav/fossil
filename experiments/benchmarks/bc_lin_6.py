from experiments.benchmarks.benchmarks_bc import six_poly
from src.shared.consts import VerifierType, LearnerType
from src.shared.activations import ActivationType
from src.shared.cegis_values import CegisConfig
from src.barrier.cegis_barrier import Cegis
from functools import partial
import traceback
import timeit
import torch


def main():

    batch_size = 1000
    system = partial(six_poly, batch_size)
    activations = [ActivationType.LINEAR]
    hidden_neurons = [10]
    opts = {CegisConfig.SYMMETRIC_BELT.k: False,
            CegisConfig.SP_HANDLE.k: True,
            CegisConfig.SP_SIMPLIFY.k: True,
            CegisConfig.ROUNDING.k: 2
            }
    try:
        start = timeit.default_timer()
        c = Cegis(6, LearnerType.NN, VerifierType.Z3, activations, system, hidden_neurons,
                  **opts)
        state, _, __, ___ = c.solve()
        end = timeit.default_timer()

        print('Elapsed Time: {}'.format(end - start))
        print("Found? {}".format(state['found']))
    except Exception as _:
        print(traceback.format_exc())


if __name__ == '__main__':
    torch.manual_seed(167)
    main()

