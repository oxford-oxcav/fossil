# Copyright (c) 2023, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import torch
import timeit
import dreal as dr
from fossil.cegis import Cegis
from fossil.consts import *
from fossil.learner import CtrlLearnerDT
from fossil.control import GeneralController
from experiments.benchmarks.benchmark_ctrl import ctrllyap_identity, ctrllyap_linear_dt
import fossil.translator as translator


class test_init(unittest.TestCase):
    def init_cegis(self):
        benchmark = ctrllyap_identity
        n_vars = 2
        system = benchmark

        # define NN parameters
        lyap_activations = [ActivationType.SQUARE]
        lyap_hidden_neurons = [2] * len(lyap_activations)

        # ctrl params
        ctrl_hidden_neurons = 2
        ctrl_inputs = 2

        opts = CegisConfig(
            N_VARS=n_vars,
            CERTIFICATE=CertificateType.LYAPUNOV,
            LLO=False,
            TIME_DOMAIN=TimeDomain.DISCRETE,
            VERIFIER=VerifierType.DREAL,
            ACTIVATION=lyap_activations,
            SYSTEM=system,
            N_HIDDEN_NEURONS=lyap_hidden_neurons,
            CTRLAYER=[ctrl_hidden_neurons, ctrl_inputs],
            CTRLACTIVATION=[ActivationType.LINEAR],
        )
        c = Cegis(opts)

        learner = c.learner
        # check type
        assert isinstance(learner, CtrlLearnerDT)
        # check features
        assert learner.layers[0].in_features == n_vars
        assert learner.layers[0].out_features == lyap_hidden_neurons[0]
        assert learner.layers[1].out_features == 1

        # check controller
        controller = c.ctrler
        assert isinstance(controller, GeneralController)
        assert controller.layers[0].in_features == n_vars
        assert controller.layers[0].out_features == ctrl_hidden_neurons
        assert controller.layers[1].out_features == ctrl_inputs

        return c

    def bigger_init(self):
        benchmark = ctrllyap_identity
        n_vars = 2
        system = benchmark

        lyap_activations = [ActivationType.SQUARE]
        lyap_hidden_neurons = [7, 17]

        # ctrl params
        ctrl_hidden_neurons = [5]
        ctrl_inputs = [2]

        opts = CegisConfig(
            N_VARS=n_vars,
            CERTIFICATE=CertificateType.LYAPUNOV,
            LLO=False,
            TIME_DOMAIN=TimeDomain.DISCRETE,
            VERIFIER=VerifierType.DREAL,
            ACTIVATION=lyap_activations,
            SYSTEM=system,
            N_HIDDEN_NEURONS=lyap_hidden_neurons,
            CTRLAYER=ctrl_hidden_neurons + ctrl_inputs,
            CTRLACTIVATION=[ActivationType.LINEAR],
        )

        c = Cegis(opts)

        learner = c.learner
        # check features
        assert learner.layers[0].in_features == n_vars
        assert learner.layers[0].out_features == lyap_hidden_neurons[0]
        assert learner.layers[1].out_features == lyap_hidden_neurons[1]
        assert learner.layers[2].out_features == 1

        # check controller
        controller = c.ctrler
        assert isinstance(controller, GeneralController)
        assert controller.layers[0].in_features == n_vars
        assert controller.layers[0].out_features == ctrl_hidden_neurons[0]
        assert controller.layers[1].out_features == ctrl_inputs[0]

        return c

    def init_unstable(self):
        benchmark = ctrllyap_linear_dt
        n_vars = 2
        system = benchmark

        # define NN parameters
        lyap_activations = [ActivationType.SQUARE]
        lyap_hidden_neurons = [2] * len(lyap_activations)

        # ctrl params
        ctrl_hidden_neurons = 2
        ctrl_inputs = 2

        opts = CegisConfig(
            N_VARS=n_vars,
            CERTIFICATE=CertificateType.LYAPUNOV,
            LLO=False,
            TIME_DOMAIN=TimeDomain.DISCRETE,
            VERIFIER=VerifierType.DREAL,
            ACTIVATION=lyap_activations,
            SYSTEM=system,
            N_HIDDEN_NEURONS=lyap_hidden_neurons,
            CTRLAYER=[ctrl_hidden_neurons, ctrl_inputs],
            CTRLACTIVATION=[ActivationType.LINEAR],
        )
        c = Cegis(opts)

        learner = c.learner
        # check type
        assert isinstance(learner, CtrlLearnerDT)
        # check features
        assert learner.layers[0].in_features == n_vars
        assert learner.layers[0].out_features == lyap_hidden_neurons[0]
        assert learner.layers[1].out_features == 1

        # check controller
        controller = c.ctrler
        assert isinstance(controller, GeneralController)
        assert controller.layers[0].in_features == n_vars
        assert controller.layers[0].out_features == ctrl_hidden_neurons
        assert controller.layers[1].out_features == ctrl_inputs

        return c

    def test_checkinit(self):
        self.init_cegis()
        self.bigger_init()
        self.init_unstable()

    def test_check_f_smt(self):
        c = self.init_cegis()
        controller = c.ctrler

        controller_in_f = c.f.controller

        assert controller == controller_in_f

        # set control weights
        # controller should be [3*x0, 8*x1]
        controller.layers[0].weight.data = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        controller.layers[1].weight.data = torch.tensor([[3.0, 0.0], [0.0, 4.0]])

        f_smt = c.f.f_smt

        xs = [dr.Variable("x0"), dr.Variable("x1")]
        # the system is
        # [x0 + u0, x1 + u1]
        # then should result
        # [x0 + 3*x0, x1 + 8*x1]

        xdots = f_smt(xs)

        assert [4.0 * xs[0], 9.0 * xs[1]] == xdots

        # another check
        # controller should be [3*x0 + 2*x1, 8*x1]
        controller.layers[0].weight.data = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        controller.layers[1].weight.data = torch.tensor([[3.0, 1.0], [0.0, 4.0]])

        f_smt = c.f.f_smt

        xs = [dr.Variable("x0"), dr.Variable("x1")]
        # the system is
        # [x0 + u0, x1 + u1]
        # then should result
        # [x0 + 3*x0 + 2*x1, x1 + 8*x1]
        xdots = f_smt(xs)

        assert [4.0 * xs[0] + 2.0 * xs[1], 9.0 * xs[1]] == xdots

    def test_correctness(self):
        c = self.init_cegis()

        controller = c.ctrler

        # controller should be [3*x0 + 2*x1, 8*x1]
        controller.layers[0].weight.data = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        controller.layers[1].weight.data = torch.tensor([[3.0, 1.0], [0.0, 4.0]])

        f_smt = c.f.f_smt

        xs = [dr.Variable("x0"), dr.Variable("x1")]
        # the system is
        # [x0 + u0, x1 + u1]
        # then should result
        # f(x) = [4*x0 + 2*x1, 9*x1]
        xdots = f_smt(xs)

        # set Lyapunov function
        # V = 3 * x0^2 + x1^2
        learner = c.learner
        learner.layers[0].weight.data = torch.tensor([[3.0, 0.0], [0.0, 1.0]])
        learner.layers[1].weight.data = torch.tensor([[1.0, 1.0]])

        # create a 'real' translator and compute V, Vdot
        regolo = translator.TranslatorDT(learner, xs, xdots, None, 1)
        res = regolo.get(**{"factors": None})
        V, Vdot = res[CegisStateKeys.V], res[CegisStateKeys.V_dot]

        # check
        desired_V = (3.0 * xs[0]) ** 2 + (1.0 * xs[1]) ** 2
        # vdot should be V(f(x)) - V(x)
        # V(f(x)) = 9 * (4*x0 + 2*x1) ** 2 + 81*x1**2
        vfx = 9.0 * (4.0 * xs[0] + 2.0 * xs[1]) ** 2 + (9.0 * xs[1]) ** 2
        desired_Vdot = vfx - desired_V

        res = dr.CheckSatisfiability(V - desired_V != 0, 0.0001)
        assert res is None
        res = dr.CheckSatisfiability(
            Vdot.Expand() - desired_Vdot.Expand() != 0.0, 0.00001
        )
        assert res is None

    def test_learning(self):
        c = self.init_unstable()
        learner = c.learner
        opt = c.optimizer
        samples = c.S
        samples_dot = c.f.f_torch(samples["lie"])
        f_torch = c.f.f_torch
        f_smt = c.f.f_smt
        xs = [dr.Variable("x0"), dr.Variable("x1")]
        print(f"Initial f(x): {f_smt(xs)[0][0]}")
        print(f"Initial f(x): {f_smt(xs)[1][0]}")
        _ = learner.learn(opt, samples, samples_dot, f_torch)

        f_smt = c.f.f_smt
        xdots = f_smt(xs)
        print(f"Final f(x): {f_smt(xs)[0][0]}")
        print(f"Final f(x): {f_smt(xs)[1][0]}")
        # check if learned something smaller than 1
        xdots[0][0] = xdots[0][0].Substitute(xs[0], 1.0)
        xdots[0][0] = xdots[0][0].Substitute(xs[1], 1.0)

        assert abs(xdots[0][0]) < 1.0

        # check if learned something smaller than 1
        xdots[1][0] = xdots[1][0].Substitute(xs[0], 1.0)
        xdots[1][0] = xdots[1][0].Substitute(xs[1], 1.0)

        assert abs(xdots[1][0]) < 1.0


if __name__ == "__main__":
    unittest.main()
