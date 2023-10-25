# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch
import timeit
import pandas as pd
from tqdm import tqdm

from experiments.benchmarks.benchmarks_lyap import *
from experiments.robustness.tqdm_redirect import std_out_err_redirect_tqdm
from fossil.cegis import Cegis

from fossil.consts import *


def test_robustness(benchmark, n_vars, domain, hidden):
    batch_size = 500

    system = partial(benchmark, batch_size)

    # define domain constraints
    outer_radius = domain
    inner_radius = 0.01

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [hidden] * len(activations)

    opts = CegisConfig(
        N_VARS=n_vars,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        CERTIFICATE=CertificateType.LYAPUNOV,
        ACTIVATION=activations,
        SYSTEM=system,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        INNER_RADIUS=inner_radius,
        OUTER_RADIUS=outer_radius,
        LLO=True,
        VERBOSE=False,
    )

    start = timeit.default_timer()
    c = Cegis(opts)
    state, vars, f_learner, iters = c.solve()
    stop = timeit.default_timer()

    return stop - start, state[CegisStateKeys.found], state["components_times"], iters


if __name__ == "__main__":
    number_of_runs = 100
    domains = [10, 20, 50, 100, 200, 500]
    hiddens = [2, 10, 50, 100]
    with std_out_err_redirect_tqdm() as orig_stdout:
        pbar = tqdm(
            total=number_of_runs * len(domains) * len(hiddens),
            file=orig_stdout,
            dynamic_ncols=True,
        )
        for domain in domains:
            for hidden in hiddens:
                res = pd.DataFrame(
                    columns=[
                        "found",
                        "iters",
                        "elapsed_time",
                        "lrn_time",
                        "reg_time",
                        "ver_time",
                        "trj_time",
                    ]
                )

                for idx in range(number_of_runs):
                    el_time, found, comp_times, iters = test_robustness(
                        benchmark=nonpoly0_lyap, n_vars=2, domain=domain, hidden=hidden
                    )
                    res = res.append(
                        {
                            "found": found,
                            "iters": iters,
                            "elapsed_time": el_time,
                            "lrn_time": comp_times[0],
                            "reg_time": comp_times[1],
                            "ver_time": comp_times[2],
                            "trj_time": comp_times[3],
                        },
                        ignore_index=True,
                    )
                    pbar.update(1)

                name_save = (
                    "robustness_lyap_domain_"
                    + str(domain)
                    + "_hdn_"
                    + str(hidden)
                    + ".csv"
                )
                res.to_csv(name_save)
