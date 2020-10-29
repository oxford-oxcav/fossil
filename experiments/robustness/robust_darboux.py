import traceback
from functools import partial

import torch
import numpy as np
import pandas as pd
import timeit

from experiments.benchmarks.benchmarks_bc import prajna07_modified, darboux
from src.barrier.cegis_barrier import Cegis
from src.shared.activations import ActivationType
from src.shared.cegis_values import CegisConfig, CegisStateKeys
from src.shared.consts import VerifierType, LearnerType
from src.plots.plot_lyap import plot_lyce


def main(h):
    batch_size = 500
    system = partial(darboux, batch_size)
    activations = [ActivationType.LINEAR, ActivationType.LIN_SQUARE_CUBIC, ActivationType.LINEAR]
    hidden_neurons = [h] * len(activations)

    start = timeit.default_timer()
    opts = {
        CegisConfig.N_VARS.k: 2,
        CegisConfig.LEARNER.k: LearnerType.NN,
        CegisConfig.VERIFIER.k: VerifierType.DREAL,
        CegisConfig.ACTIVATION.k: activations,
        CegisConfig.SYSTEM.k: system,
        CegisConfig.N_HIDDEN_NEURONS.k: hidden_neurons,
        CegisConfig.SP_SIMPLIFY.k: True,
    }
    c = Cegis(**opts)
    state, vars, f_learner, iters = c.solve()
    end = timeit.default_timer()

    print('Elapsed Time: {}'.format(end - start))
    print("Found? {}".format(state[CegisStateKeys.found]))

    return end-start, state[CegisStateKeys.found], state['components_times'], iters


if __name__ == '__main__':
    number_of_runs = 10

    for hidden in [100]:
        res = pd.DataFrame(columns=['found_bc', 'iters', 'elapsed_time',
                                    'lrn_time', 'reg_time', 'ver_time', 'trj_time'])

        for idx in range(number_of_runs):
            el_time, found, comp_times, iters = main(h=hidden)
            res = res.append({'found_bc': found, 'iters': iters,
                              'elapsed_time': el_time,
                              'lrn_time': comp_times[0], 'reg_time': comp_times[1],
                              'ver_time': comp_times[2], 'trj_time': comp_times[3]},
                             ignore_index=True)

        name_save = 'darboux' + '_hdn_' + str(hidden) + '_1st_run.csv'
        res.to_csv(name_save)


