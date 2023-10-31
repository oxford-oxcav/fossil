# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import mock
import z3
import dreal as dr
import fossil.learner as learner
import fossil.translator as translator
from fossil.consts import ActivationType, CegisStateKeys, CegisConfig
from experiments.benchmarks.benchmarks_lyap import poly_2
from experiments.benchmarks.models import DebugDT
from fossil.learner import LearnerDT, LearnerCT
import torch
from unittest.mock import MagicMock


class TranslatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.n_vars = 2
        system = poly_2
        self.f, _, self.S_d, _ = system()
        self.f_learner = self.f
        self.f_verifier = self.f
        self.hidden = [3]
        self.activate = [ActivationType.SQUARE]
        self.x = [z3.Real("x"), z3.Real("y")]
        self.xdot = self.f(self.x)
        self.learn_method = MagicMock(return_value=None)

    # given a point, the consolidator returns a list of points - trajectory -
    # that lead towards the max of Vdot
    def test_fromNet_returnSimplifiedVAndVdot(self):
        # give a value to a hypothetical cex
        torch.set_default_dtype(torch.float64)
        point = torch.tensor([1.0, 2.0])
        point.requires_grad = True

        # def neural learner
        with mock.patch.object(learner.LearnerCT, "learn") as lrner:
            # setup lrner
            lrner.input_size = 2
            lrner.acts = [ActivationType.SQUARE]
            lrner.layers = [
                torch.nn.Linear(2, 3, bias=False),
                torch.nn.Linear(3, 1, bias=False),
            ]
            lrner.layers[0].weight = torch.nn.Parameter(
                torch.tensor([[1.234, 0.0], [0.0, 1.234], [0.0, 0.0]])
            )
            lrner.layers[1].weight = torch.nn.Parameter(
                torch.tensor([1.0, 1.0, 1.0]).reshape(1, 3)
            )

            # create a 'real' translator and compute V, Vdot
            regolo = translator.TranslatorCT(self.x, self.xdot, 1, CegisConfig())
            res = regolo.get(**{"net": lrner})
            V, Vdot = res[CegisStateKeys.V], res[CegisStateKeys.V_dot]

            # given the benchamrk, the NN and the rounding, the correct expr of V and Vdot are
            # V = (1.2*x)**2 + (1.2*y)**2 = 1.44 * x**2 + 1.44 * y**2
            # Vdot = 2 * 1.44 * x * (- x**3 + y) + 2 * 1.44 * y * (- x - y)
            desired_V = (1.2 * self.x[0]) ** 2 + (1.2 * self.x[1]) ** 2
            desired_Vdot = (
                2 * 1.44 * self.x[0] * self.xdot[0]
                + 2 * 1.44 * self.x[1] * self.xdot[1]
            )

            self.assertEqual(z3.simplify(V), z3.simplify(desired_V))
            self.assertEqual(z3.simplify(Vdot), z3.simplify(desired_Vdot))

            # check that Vdot(trajectory) is an increasing sequence
    def test_Translator_with_bias(self):

        system = DebugDT()
        n_vars = DebugDT().n_vars

        learner = LearnerCT(n_vars, self.learn_method, *[3],
                            activation=[ActivationType.SQUARE],
                            bias=True, config=CegisConfig())

        learner.layers = [
            torch.nn.Linear(2, 3, bias=True),
            torch.nn.Linear(3, 1, bias=True),
        ]
        learner.layers[0].weight = torch.nn.Parameter(
            torch.tensor([[1.234, 0.0], [0.0, 1.234], [0.0, 0.0]])
        )
        learner.layers[0].bias = torch.nn.Parameter(
            torch.tensor([[0.], [1.], [0.]])[:, 0]
        )
        learner.layers[1].weight = torch.nn.Parameter(
            torch.tensor([1.0, 1.0, 1.0]).reshape(1, 3)
        )
        learner.layers[1].bias = torch.nn.Parameter(
            torch.tensor([[7.]])[:, 0]
        )

        # create a translator and compute V, Vdot
        self.x = [dr.Variable("x"), dr.Variable("y")]
        self.xdot = system.f_smt(self.x)

        regolo = translator.TranslatorCT(self.x, self.xdot, 1, CegisConfig())
        res = regolo.get(**{"net": learner})
        V, Vdot = res[CegisStateKeys.V], res[CegisStateKeys.V_dot]

        # given the benchmark, the NN and the rounding, the correct expr of C and Cdot are
        # C = (1.2*x)**2 + (1.2*y + 1)**2 + 7 = 1.44*x**2 + 2.4*y + 1.44* y**2 + 8
        # Cdot = gradC * f(x) where f(x) = [0.5*y, -0.1*x]
        # = 2 * 1.44 * x * 0.5*y + (2*1.44*y + 2.4) * (-0.1*x)
        # = -0.24 x + 1.152 x y
        desired_V = (1.2 * self.x[0]) ** 2 + (1.2 * self.x[1] + 1.) ** 2 + 7.
        desired_Vdot = 2 * 1.44 * self.x[0] * self.xdot[0] + (2*1.44*self.x[1] + 2.4) * self.xdot[1]

        # dreal will return slightly different expression, due to numerical manipulation of float numbers.
        # so pick random values for x and y, and evaluate the difference between the two expression
        # if the difference is < 1e-5, then we say that's the same expression
        for idx in range(1000):
            x_sample, y_sample = float(torch.randn((1,))), float(torch.randn((1,)))
            diff_c = ((V - desired_V).Substitute({self.x[0]:x_sample, self.x[1]: y_sample})).Evaluate()
            diff_cdot = ((Vdot - desired_Vdot).Substitute({self.x[0]:x_sample, self.x[1]: y_sample})).Evaluate()
            # print(f'{diff_c}, {diff_cdot}')
            self.assertTrue(abs(diff_c) < 1e-5)
            self.assertTrue(abs(diff_cdot) < 1e-5)


class TranslatorDiscreteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.n_vars = 2
        system = poly_2
        self.f, _, self.S_d, _ = system()
        self.f_learner = self.f
        self.f_verifier = self.f
        self.hidden = [3]
        self.activate = [ActivationType.SQUARE]
        self.x = [z3.Real("x"), z3.Real("y")]
        self.xdot = self.f(self.x)
        self.learn_method = MagicMock(return_value=None)

    # given a point, the consolidator returns a list of points - trajectory -
    # that lead towards the max of Vdot
    def test_fromNet_returnSimplifiedVAndVdot(self):
        # give a value to a hypothetical cex
        torch.set_default_dtype(torch.float64)
        point = torch.tensor([1.0, 2.0])
        point.requires_grad = True

        # def neural learner
        with mock.patch.object(learner.LearnerDT, "learn") as lrner:
            # setup lrner
            lrner.input_size = 2
            lrner.acts = [ActivationType.SQUARE]
            lrner.layers = [
                torch.nn.Linear(2, 3, bias=False),
                torch.nn.Linear(3, 1, bias=False),
            ]
            lrner.layers[0].weight = torch.nn.Parameter(
                torch.tensor([[1.234, 0.0], [0.0, 1.234], [0.0, 0.0]])
            )
            lrner.layers[1].weight = torch.nn.Parameter(
                torch.tensor([1.0, 1.0, 1.0]).reshape(1, 3)
            )

            # create a 'real' translator and compute V, Vdot
            regolo = translator.TranslatorDT(self.x, self.xdot, 1, CegisConfig())
            res = regolo.get(**{"net": lrner})
            V, Vdot = res[CegisStateKeys.V], res[CegisStateKeys.V_dot]

            # given the benchamrk, the NN and the rounding, the correct expr of V and Vdot are
            # V = (1.2*x)**2 + (1.2*y)**2 = 1.44 * x**2 + 1.44 * y**2
            # Vdot = 2 * 1.44 * x * (- x**3 + y) + 2 * 1.44 * y * (- x - y)
            desired_V = (1.2 * self.x[0]) ** 2 + (1.2 * self.x[1]) ** 2
            desired_Vdot = (
                (1.2 * self.xdot[0]) ** 2 + (1.2 * self.xdot[1]) ** 2 - desired_V
            )
            self.assertEqual(z3.simplify(V), z3.simplify(desired_V))
            self.assertEqual(z3.simplify(Vdot), z3.simplify(desired_Vdot))

    def test_Translator_with_bias(self):

        system = DebugDT()
        n_vars = DebugDT().n_vars

        learner = LearnerDT(n_vars, self.learn_method, *[3],
                            activation=[ActivationType.SQUARE],
                            bias=True, config=CegisConfig())

        learner.layers = [
            torch.nn.Linear(2, 3, bias=True),
            torch.nn.Linear(3, 1, bias=True),
        ]
        learner.layers[0].weight = torch.nn.Parameter(
            torch.tensor([[1.234, 0.0], [0.0, 1.234], [0.0, 0.0]])
        )
        learner.layers[0].bias = torch.nn.Parameter(
            torch.tensor([[0.], [1.], [0.]])[:, 0]
        )
        learner.layers[1].weight = torch.nn.Parameter(
            torch.tensor([1.0, 1.0, 1.0]).reshape(1, 3)
        )
        learner.layers[1].bias = torch.nn.Parameter(
            torch.tensor([[7.]])[:, 0]
        )

        # create a translator and compute V, Vdot
        self.x = [dr.Variable("x"), dr.Variable("y")]
        self.xdot = system.f_smt(self.x)

        regolo = translator.TranslatorDT(self.x, self.xdot, 1, CegisConfig())
        res = regolo.get(**{"net": learner})
        V, Vdot = res[CegisStateKeys.V], res[CegisStateKeys.V_dot]

        # given the benchmark, the NN and the rounding, the correct expr of C and Cdot are
        # C = (1.2*x)**2 + (1.2*y + 1)**2 + 7 = 1.44*x**2 + 2.4*y + 1.44* y**2 + 8
        # Cdot = C(f(x)) - C(x) where f(x) = [0.5*y, -0.1*x]
        # = (1.2 * 0.5 * y)**2 + (1.2 * (-0.1)*x + 1)**2 + 7 - (1.44 * x**2 + (1.2 * y**2+1) + 7)
        # = -0.24* x - 1.4256 *x**2 - 0.84 *y**2
        desired_V = (1.2 * self.x[0]) ** 2 + (1.2 * self.x[1] + 1.) ** 2 + 7.
        desired_Vdot = (
                (1.2 * self.xdot[0]) ** 2 + (1.2 * self.xdot[1] + 1.) ** 2 + 7. - desired_V
        )

        # dreal will return slightly different expression, due to numerical manipulation of float numbers.
        # so pick random values for x and y, and evaluate the difference between the two expression
        # if the difference is < 1e-5, then we say that's the same expression
        for idx in range(1000):
            x_sample, y_sample = float(torch.randn((1,))), float(torch.randn((1,)))
            diff_c = ((V - desired_V).Substitute({self.x[0]:x_sample, self.x[1]: y_sample})).Evaluate()
            diff_cdot = ((Vdot - desired_Vdot).Substitute({self.x[0]:x_sample, self.x[1]: y_sample})).Evaluate()
            # print(f'{diff_c}, {diff_cdot}')
            self.assertTrue(abs(diff_c) < 1e-5)
            self.assertTrue(abs(diff_cdot) < 1e-5)

    def test_Translator_with_bias_learner_vs_verifier(self):

        system = DebugDT()
        n_vars = DebugDT().n_vars

        # random weights
        learner = LearnerDT(n_vars, self.learn_method, *[3],
                            activation=[ActivationType.SQUARE],
                            bias=True, config=CegisConfig())

        # create a translator and compute V, Vdot
        self.x = [dr.Variable("x"), dr.Variable("y")]
        self.xdot = system.f_smt(self.x)

        regolo = translator.TranslatorDT(self.x, self.xdot, -1, CegisConfig())
        res = regolo.get(**{"net": learner})
        V, Vdot = res[CegisStateKeys.V], res[CegisStateKeys.V_dot]

        # dreal will return slightly different expression, due to numerical manipulation of float numbers.
        # so pick random values for x and y, and evaluate the difference between the two expression
        # if the difference is < 1e-5, then we say that's the same expression
        for idx in range(1000):
            x_sample, y_sample = float(torch.randn((1,))), float(torch.randn((1,)))
            sample = torch.tensor([[x_sample, y_sample]])
            f_sample = torch.stack(system.f_torch(torch.tensor([[x_sample, y_sample]]))).T

            c_lrn, cdot_lrn, _ = learner.get_all(sample, f_sample)

            diff_c = ((V - c_lrn).Substitute({self.x[0]:x_sample, self.x[1]: y_sample})).Evaluate()
            diff_cdot = ((Vdot - cdot_lrn).Substitute({self.x[0]:x_sample, self.x[1]: y_sample})).Evaluate()
            # print(f'{diff_c}, {diff_cdot}')
            self.assertTrue(abs(diff_c) < 1e-5)
            self.assertTrue(abs(diff_cdot) < 1e-5)


if __name__ == "__main__":
    unittest.main()
