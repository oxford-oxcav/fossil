import torch
import timeit
import pandas as pd
from src.lyap.cegis_lyap import Cegis
from experiments.benchmarks.benchmarks_lyap import *
from src.shared.activations import ActivationType
from src.shared.consts import VerifierType, LearnerType
from src.shared.cegis_values import CegisConfig
from src.plots.plot_lyap import plot_lyce
from functools import partial


def test_lnn(benchmark, n_vars):
    batch_size = 500

    system = partial(benchmark, batch_size)

    # define domain constraints
    outer_radius = 500
    inner_radius = 0.01

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [2] * len(activations)

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
    # if len(vars) == 2 and state['found']:
    #     plot_lyce(np.array(vars), state['V'],
    #                   state['V_dot'], f_learner)

    print('Elapsed Time: {}'.format(stop-start))

    return stop-start, state['found'], state['computational_times']


if __name__ == '__main__':
    res = pd.DataFrame(columns=['elapsed_time', 'found_lyapunov',
                                'lrn_time', 'reg_time', 'ver_time', 'trj_time'])
    number_of_runs = 20
    for idx in range(number_of_runs):
        el_time, found, comp_times = test_lnn(benchmark=nonpoly0, n_vars=2)
        res = res.append({'elapsed_time': el_time, 'found_lyapunov': found,
                          'lrn_time': comp_times[0], 'reg_time': comp_times[1],
                          'ver_time': comp_times[2], 'trj_time': comp_times[3]},
                         ignore_index=True)

    res.to_csv('res_dom_500_hdn_002.csv')
    print(res)
    print('Avg results: \n', res.mean())
    print('Components results: \n', res.max()/number_of_runs)
    print('Max results: \n', res.max())
    print('Min results: \n', res.min())
    success_res = res.loc[res['found_lyapunov']==True]
    print('Avg success results: \n', success_res.mean())
    print('Max success results: \n', success_res.max())
    print('Number of Found and fail: \n', res.found_lyapunov.value_counts())
