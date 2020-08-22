import traceback
from functools import partial

import torch
import timeit

from experiments.benchmarks import obstacle_avoidance
from barrier.cegis import Cegis
from barrier.activations import ActivationType
from barrier.consts import VerifierType, LearnerType


def main():
    MIN_TO_SEC = 60
    batch_size = 500
    system = partial(obstacle_avoidance, batch_size)
    activations = [ActivationType.LIN_SQUARE]
    hidden_neurons = [10]
    try:
        start = timeit.default_timer()
        c = Cegis(3, LearnerType.NN, VerifierType.DREAL, activations, system, hidden_neurons,
                  sp_simplify=False, cegis_time=30 * MIN_TO_SEC)
        _, found, _ = c.solve()
        end = timeit.default_timer()

        print('Elapsed Time: {}'.format(end - start))
        print("Found? {}".format(found))
    except Exception as _:
        print(traceback.format_exc())


if __name__ == '__main__':
    torch.manual_seed(167)
    main()
