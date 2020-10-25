import torch
import timeit
import pandas as pd
from src.lyap.cegis_lyap import Cegis
from experiments.benchmarks.benchmarks_lyap import *
from src.shared.activations import ActivationType
from src.shared.consts import VerifierType, LearnerType
from src.shared.cegis_values import CegisConfig
from src.shared.utils import print_section
from functools import partial


def test_lnn(benchmark, n_vars, domain, hidden):
    batch_size = 500

    system = partial(benchmark, batch_size)

    # define domain constraints
    outer_radius = domain
    inner_radius = 0.01

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [hidden] * len(activations)

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
    }

    start = timeit.default_timer()
    c = Cegis(**opts)
    state, vars, f_learner, iters = c.solve()
    stop = timeit.default_timer()

    # plotting -- only for 2-d systems
    # if len(vars) == 2 and state['found']:
    #     plot_lyce(np.array(vars), state['V'],
    #                   state['V_dot'], f_learner)

    # print('Elapsed Time: {}'.format(stop-start))

    return stop-start, state['found'], state['components_times'], iters


def result_analysis(df, n):

    print('Avg Time: ', df.mean()['elapsed_time'])
    print('Learner avg: ', df.max()['lrn_time']/n)
    print('Regulariser avg: ', df.max()['reg_time'] / n)
    print('Verifier avg: ', df.max()['ver_time'] / n)
    print('Trajectoriser avg: ', df.max()['trj_time'] / n)
    print('Max Time: ', df.max()['elapsed_time'])
    print('Min Time: ', df.min()['elapsed_time'])

    success_res = df.loc[df['found_lyapunov']==True]
    print('Avg success results: ', success_res.mean()['elapsed_time'])
    print('Max success results: ', success_res.max()['elapsed_time'])
    print('Number of Found and Fail: ', df.found_lyapunov.value_counts())


if __name__ == '__main__':
    number_of_runs = 25

    for domain in [20]:  # 10, 20, 50, 100, 200, 500]:
        for hidden in [2, 10, 50, 100]:
            res = pd.DataFrame(columns=['found_lyapunov', 'iters', 'elapsed_time',
                                        'lrn_time', 'reg_time', 'ver_time', 'trj_time'])

            for idx in range(number_of_runs):
                el_time, found, comp_times, iters = test_lnn(benchmark=nonpoly0, n_vars=2,
                                                             domain=domain, hidden=hidden)
                res = res.append({'found_lyapunov': found, 'iters': iters,
                                  'elapsed_time': el_time,
                                  'lrn_time': comp_times[0], 'reg_time': comp_times[1],
                                  'ver_time': comp_times[2], 'trj_time': comp_times[3]},
                                 ignore_index=True)

            name_save = 'dreal_dom_'+str(domain)+'_hdn_'+str(hidden)+'_4th_run.csv'
            res.to_csv(name_save)
    # print_section('Result', 'Analysis')

    # result_analysis(res, number_of_runs)

