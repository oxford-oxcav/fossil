# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
import unittest
from unittest import mock
import z3
import src.learner as learner
import src.translator as translator
from src.shared.activations import ActivationType
from src.shared.cegis_values import CegisStateKeys
from experiments.benchmarks.benchmarks_lyap import poly_2
import torch


class TranslatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.n_vars = 2
        system = poly_2
        self.f, _, self.S_d, _ = system()
        self.f_learner = self.f
        self.f_verifier = self.f
        self.hidden = [3]
        self.activate = [ActivationType.SQUARE]
        self.x = [z3.Real('x'), z3.Real('y')]
        self.xdot = self.f(self.x)

    # given a point, the consolidator returns a list of points - trajectory -
    # that lead towards the max of Vdot
    def test_fromNet_returnSimplifiedVAndVdot(self):
        # give a value to a hypothetical cex
        torch.set_default_dtype(torch.float64)
        point = torch.tensor([1., 2.])
        point.requires_grad = True

        # def neural learner
        with mock.patch.object(learner.LearnerCT, 'learn') as lrner:
            # setup lrner
            lrner.input_size = 2
            lrner.acts = [ActivationType.SQUARE]
            lrner.layers = [
                torch.nn.Linear(2, 3, bias=False),
                torch.nn.Linear(3, 1, bias=False)
            ]
            lrner.layers[0].weight = torch.nn.Parameter(torch.tensor(
                [[1.234, 0.0],
                 [0.0, 1.234],
                 [0.0, 0.0]
            ]))
            lrner.layers[1].weight = torch.nn.Parameter(
                torch.tensor([1.0, 1.0, 1.0]).reshape(1, 3)
            )

            # create a 'real' translator and compute V, Vdot
            regolo = translator.TranslatorCT(lrner, self.x, self.xdot, None, 1)
            res = regolo.get(**{'factors': None})
            V, Vdot = res[CegisStateKeys.V], res[CegisStateKeys.V_dot]

            # given the benchamrk, the NN and the rounding, the correct expr of V and Vdot are
            # V = (1.2*x)**2 + (1.2*y)**2 = 1.44 * x**2 + 1.44 * y**2
            # Vdot = 2 * 1.44 * x * (- x**3 + y) + 2 * 1.44 * y * (- x - y)
            desired_V = (1.2 * self.x[0])**2 + (1.2 * self.x[1])**2
            desired_Vdot = 2 * 1.44 * self.x[0] * self.xdot[0] \
                           + 2 * 1.44 * self.x[1] * self.xdot[1]

            self.assertEqual(z3.simplify(V), z3.simplify(desired_V))
            self.assertEqual(z3.simplify(Vdot), z3.simplify(desired_Vdot))

            # check that Vdot(trajectory) is an increasing sequence

class TranslatorDiscreteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.n_vars = 2
        system = poly_2
        self.f, _, self.S_d, _ = system()
        self.f_learner = self.f
        self.f_verifier = self.f
        self.hidden = [3]
        self.activate = [ActivationType.SQUARE]
        self.x = [z3.Real('x'), z3.Real('y')]
        self.xdot = self.f(self.x)

    # given a point, the consolidator returns a list of points - trajectory -
    # that lead towards the max of Vdot
    def test_fromNet_returnSimplifiedVAndVdot(self):
        # give a value to a hypothetical cex
        torch.set_default_dtype(torch.float64)
        point = torch.tensor([1., 2.])
        point.requires_grad = True

        # def neural learner
        with mock.patch.object(learner.LearnerDT, 'learn') as lrner:
            # setup lrner
            lrner.input_size = 2
            lrner.acts = [ActivationType.SQUARE]
            lrner.layers = [
                torch.nn.Linear(2, 3, bias=False),
                torch.nn.Linear(3, 1, bias=False)
            ]
            lrner.layers[0].weight = torch.nn.Parameter(torch.tensor(
                [[1.234, 0.0],
                 [0.0, 1.234],
                 [0.0, 0.0]
            ]))
            lrner.layers[1].weight = torch.nn.Parameter(
                torch.tensor([1.0, 1.0, 1.0]).reshape(1, 3)
            )

            # create a 'real' translator and compute V, Vdot
            regolo = translator.TranslatorDT(lrner, self.x, self.xdot, None, 1)
            res = regolo.get(**{'factors': None})
            V, Vdot = res[CegisStateKeys.V], res[CegisStateKeys.V_dot]

            # given the benchamrk, the NN and the rounding, the correct expr of V and Vdot are
            # V = (1.2*x)**2 + (1.2*y)**2 = 1.44 * x**2 + 1.44 * y**2
            # Vdot = 2 * 1.44 * x * (- x**3 + y) + 2 * 1.44 * y * (- x - y)
            desired_V = (1.2 * self.x[0])**2 + (1.2 * self.x[1])**2
            desired_Vdot = (1.2 * self.xdot[0])**2 + (1.2 * self.xdot[1])**2 - desired_V
            self.assertEqual(z3.simplify(V), z3.simplify(desired_V))
            self.assertEqual(z3.simplify(Vdot), z3.simplify(desired_Vdot))




if __name__ == '__main__':
    unittest.main()