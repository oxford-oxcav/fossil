import traceback
from functools import partial

import torch
import timeit

from experiments.benchmarks.benchmarks_bc import elementary
from src.barrier.cegis_barrier import Cegis
from src.shared.activations import ActivationType
from src.shared.consts import VerifierType, LearnerType


def main():
    MIN_IN_SEC = 60
    batch_size = 500
    system = partial(elementary, batch_size)
    activations = [ActivationType.LIN_SQUARE]
    hidden_neurons = [10]
    try:
        start = timeit.default_timer()
        c = Cegis(2, LearnerType.NN, VerifierType.DREAL, activations, system, hidden_neurons,
                  sp_simplify=True, cegis_time=30 * MIN_IN_SEC)
        _, found, _ = c.solve()
        end = timeit.default_timer()

        print('Elapsed Time: {}'.format(end - start))
        print("Found? {}".format(found))
    except Exception as _:
        print(traceback.format_exc())


if __name__ == '__main__':
    torch.manual_seed(167)
    main()
