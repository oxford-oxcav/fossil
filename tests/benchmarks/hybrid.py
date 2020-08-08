import traceback
from functools import partial

import torch
import timeit

from tests.benchmarks import twod_hybrid
from barrier.cegis import Cegis
from barrier.activations import ActivationType
from barrier.consts import VerifierType, LearnerType


def main():
    MIN_TO_SEC = 60
    batch_size = 500
    system = partial(twod_hybrid, batch_size)
    activations = [ActivationType.LIN_SQUARE]
    hidden_neurons = [3] * len(activations)
    try:
        start = timeit.default_timer()
        c = Cegis(2, LearnerType.NN, VerifierType.Z3, activations, system, hidden_neurons,
                  sp_simplify=False, cegis_time=30 * MIN_TO_SEC, sp_handle=False, symmetric_belt=True)
        _, found, _ = c.solve()
        end = timeit.default_timer()

        print('Elapsed Time: {}'.format(end - start))
        print("Found? {}".format(found))
    except Exception as _:
        print(traceback.format_exc())


if __name__ == '__main__':
    torch.manual_seed(167)
    main()