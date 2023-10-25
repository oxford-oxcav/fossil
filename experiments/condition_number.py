# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import timeit

# pylint: disable=not-callable
import torch

import fossil.domains as sets
import fossil.plotting as plotting
from experiments.benchmarks import models
from fossil.cegis import Cegis
from fossil.consts import *

global lam
LAM = (
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    1.5,
    2,
    2.5,
    3,
    0.075,
    0.05,
    0.03,
    0.025,
)
VER = (VerifierType.DREAL, VerifierType.Z3)


class LinearCondition(models.CTModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return torch.stack([-lam * x, -y]).T

    def f_smt(self, v):
        x, y = v
        return [-lam * x, -y]


class LinearConditionControl(models.ControllableCTModel):
    n_vars = 2
    n_u = 1

    def f_torch(self, v, u):
        x, y = v[:, 0], v[:, 1]
        u1 = u[:, 0]
        return torch.stack([lam * x + u1, -y]).T

    def f_smt(self, v, u):
        x, y = v
        u1 = u[0, 0]
        return [lam * x + u1, -y]


def test_lnn():
    ###########################################
    ###
    #############################################
    n_vars = 2

    ol_system = LinearConditionControl()
    system = lambda ctrl: models.GeneralClosedLoopModel(ol_system, ctrl)

    # XD = sets.Rectangle([-1.5, -1.5], [1.5, 1.5])
    XD = sets.Torus([0, 0], 1.5, 0.01)
    XS = sets.Rectangle([-1, -1], [1, 1])
    XI = sets.Rectangle([-0.5, -0.5], [0.5, 0.5])
    XG = sets.Rectangle([-0.1, -0.1], [0.1, 0.1])

    SU = sets.SetMinus(XD, XS)  # Data for unsafe set
    SD = sets.SetMinus(XS, XG)  # Data for lie set

    D = {
        "lie": XD,
        "init": XI,
        "safe": XS,
        "goal": XG,
    }
    symbolic_domains = {
        "lie": XD.generate_domain,
        "init": XI.generate_domain,
        "safe_border": XS.generate_boundary,
        "safe": XS.generate_domain,
        "goal": XG.generate_domain,
    }
    data = {
        "lie": XD.generate_data(1000),
        "init": XI.generate_data(1000),
        "unsafe": SU.generate_data(1000),
    }
    F = lambda *args: (system(*args), symbolic_domains, data, sets.inf_bounds_n(2))

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [12] * len(activations)

    opts = CegisConfig(
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.RWS,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        SYSTEM=F,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        CEGIS_MAX_ITERS=10,
        VERBOSE=False,
        CTRLAYER=[8, 1],
        CTRLACTIVATION=[ActivationType.LINEAR],
    )

    start = timeit.default_timer()
    c = Cegis(opts)
    state, _, _, _ = c.solve()
    stop = timeit.default_timer()
    print("Elapsed Time: {}".format(stop - start))

    # plotting.benchmark(
    #     c.f,
    #     c.learner,
    #     D,
    #     xrange=[-1.1, 1.1],
    #     yrange=[-1.1, 1.1],
    #     levels=[0],
    # )
    return state["found"], stop - start


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    torch.manual_seed(167)
    torch.set_num_threads(1)
    res_dreal = []
    res_z3 = []
    T_dreal = []
    T_z3 = []
    res_count = 0
    N_reps = 10
    LAM = sorted(LAM)
    condition_num = [max(1.0, l) / min(1.0, l) for l in LAM]
    for v in VER:
        for l in LAM:
            T = 0
            lam = l
            for i in range(N_reps):
                res, t = test_lnn()
                T += t
                if res:
                    res_count += 1
            if v == VerifierType.DREAL:
                res_dreal.append(res_count / N_reps)
                T_dreal.append(T / N_reps)
            else:
                res_z3.append(res_count / N_reps)
                T_z3.append(T / N_reps)
            res_count = 0
    fig, ax1 = plt.subplots()
    ax1.plot(
        condition_num,
        res_z3,
        label="Z3-success",
        marker="x",
        color="tab:orange",
        linestyle="-.",
    )
    ax1.plot(
        condition_num,
        res_dreal,
        label="dReal-success",
        marker="x",
        color="tab:blue",
        linestyle="--",
    )
    ax1.tick_params(axis="y")
    ax2 = ax1.twinx()
    ax2.plot(condition_num, T_z3, label="Z3-Time", marker=".", color="tab:orange")
    ax2.plot(condition_num, T_dreal, label="dReal-Time", marker=".", color="tab:blue")
    ax2.tick_params(axis="y")
    ax1.legend(loc="center right")
    ax2.legend(loc="center left")
    ax1.set_xlabel("Conditon Number")
    ax1.set_ylabel("Success Rate ({} runs)".format(N_reps))
    ax2.set_ylabel("Average Time (s)")
    ax1.set_xscale("log")
    plt.savefig("Condition-Success-CTRLRWS-reg01.pdf", bbox_inches="tight")
