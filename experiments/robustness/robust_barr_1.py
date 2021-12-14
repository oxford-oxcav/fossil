# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
import traceback
from functools import partial

import torch
import numpy as np
import pandas as pd
import timeit
from tqdm import tqdm

from experiments.benchmarks.benchmarks_bc import barr_1
from experiments.robustness.tqdm_redirect import std_out_err_redirect_tqdm
from src.shared.components.cegis import Cegis
from src.shared.activations import ActivationType
from src.shared.cegis_values import CegisConfig, CegisStateKeys
from src.shared.consts import VerifierType, TimeDomain, CertificateType


def test_robustness(h):
    batch_size = 500
    system = partial(barr_1, batch_size)
    activations = [ActivationType.LINEAR]
    hidden_neurons = [h] * len(activations)
    
    start = timeit.default_timer()
    opts = {
        CegisConfig.N_VARS.k: 2,
        CegisConfig.TIME_DOMAIN.k: TimeDomain.CONTINUOUS,
        CegisConfig.VERIFIER.k: VerifierType.DREAL,
        CegisConfig.CERTIFICATE.k: CertificateType.BARRIER,
        CegisConfig.ACTIVATION.k: activations,
        CegisConfig.SYSTEM.k: system,
        CegisConfig.N_HIDDEN_NEURONS.k: hidden_neurons,
        CegisConfig.SP_SIMPLIFY.k: True,
        CegisConfig.VERBOSE.k: False
    }
    c = Cegis(**opts)
    state, vars, f_learner, iters = c.solve()
    end = timeit.default_timer()

    return end-start, state[CegisStateKeys.found], state['components_times'], iters


if __name__ == '__main__':
    number_of_runs = 100
    hidden_neurons = [2, 10, 50, 100]
    with std_out_err_redirect_tqdm() as orig_stdout:
        pbar = tqdm(total=number_of_runs*len(hidden_neurons), file=orig_stdout, dynamic_ncols=True)
        for hidden in hidden_neurons:
            res = pd.DataFrame(columns=['found', 'iters', 'elapsed_time',
                                        'lrn_time', 'reg_time', 'ver_time', 'trj_time'])

            for idx in range(number_of_runs):
                el_time, found, comp_times, iters = test_robustness(h=hidden)
                res = res.append({'found': found, 'iters': iters,
                                'elapsed_time': el_time,
                                'lrn_time': comp_times[0], 'reg_time': comp_times[1],
                                'ver_time': comp_times[2], 'trj_time': comp_times[3]},
                                ignore_index=True)
                pbar.update(1)
            name_save = 'robustness_barrier' + '_hdn_' + str(hidden) + '.csv'
            res.to_csv(name_save)


