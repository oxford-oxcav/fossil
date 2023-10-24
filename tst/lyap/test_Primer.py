# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import numpy as np
import sympy as sp
import torch

from fossil.primer import Primer
from fossil.system import LinearSystem, NonlinearSystem
from fossil.consts import PrimerMode


class TestPrimer(unittest.TestCase):
    def setUp(self):
        self.v = sp.symbols("x0 x1", real=True)
        x0, x1 = self.v
        f1 = [-(x0**3) + x1, -x0 - x1]
        f2 = [x1, x0 - x0**3 - 2 * x1 + x0**2 * x1]
        f3 = [-x0 - 3, -x1 - 2]
        self.sys1 = Primer.create_Primer(f=f1, r=10, mode=PrimerMode.LYAPUNOV)
        self.sys2 = Primer.create_Primer(f=f2, r=10, mode=PrimerMode.LYAPUNOV)
        self.sys3 = Primer.create_Primer(f=f3, r=10, mode=PrimerMode.LYAPUNOV)

    def test_check_input(self):
        inputs = ["1", "5", "1.4", "a", "[1,2]"]
        returns = [False, True, True, True, True]
        for iii, choice in enumerate(inputs):
            if returns[iii]:
                self.assertTrue(self.sys1.check_input(choice))
            else:
                self.assertFalse(self.sys1.check_input(choice))

    def test_get_shift(self):
        self.sys1.get_shift()
        # self.sys2.get_shift()
        self.sys3.get_shift()
        f1 = self.sys1.evaluate_dynamics
        # f2 = self.sys2.evaluate_dynamics
        f3 = self.sys3.evaluate_dynamics
        self.assertSequenceEqual(f1(torch.tensor([0, 0]).T), [0, 0])
        # self.assertSequenceEqual(f2(torch.tensor([0,0]).T), [0,0])
        self.assertSequenceEqual(f3(torch.tensor([0, 0]).T), [0, 0])

    def test_validate_eqbm_input(self):
        eqbm_true = sp.sympify([-3, -2])
        eqbm_false = sp.sympify([0, 0])
        self.assertTrue(self.sys3.validate_eqbm_input(eqbm_true))
        self.assertFalse(self.sys3.validate_eqbm_input(eqbm_false))


class TestLinearSystem(unittest.TestCase):
    def test_instantiation(
        self,
    ):
        A = np.array([[1, 4], [2, -3]])
        f = LinearSystem(A)
        X = np.array([[0, 0], [3, 4], [-2, -3], [-4, 6]])
        Xdot_true = np.array([[0, 0], [19, -6], [-14, 5], [20, -26]]).T
        Xdot = []

        for x in X:
            Xdot.append(f.evaluate_f(x))
        self.assertSequenceEqual(np.array(Xdot).T.tolist(), Xdot_true.tolist())


class NonlinearSystemTest(unittest.TestCase):
    def setUp(self):
        self.v = sp.symbols("x0 x1", real=True)
        x0, x1 = self.v
        f = [-(x0**3) - x1**2, x0 * x1 - x1**3]
        self.sys = NonlinearSystem(f)

    def test_find_equilibria(self):
        # How do you test this?
        x = [(0.0, 0.0)]
        self.assertSequenceEqual(x, self.sys.equilibria)

    def test_get_Jacobian(self):  #
        x0, x1 = self.v
        J = [[-3 * x0**2, -2 * x1], [x1, x0 - 3 * x1**2]]
        self.assertSequenceEqual(J, self.sys.jacobian.tolist())

    def test_evaluate_Jacobian(self):
        x = [5, -4]
        J_x = np.array([[-75.0, 8.0], [-4.0, -43.0]])
        self.assertSequenceEqual(J_x.tolist(), self.sys.evaluate_Jacobian(x).tolist())

    def test_check_stability(self):
        A_s = np.array([[-75.0, 8.0], [-4.0, -43.0]])
        A_u = np.array([[5.0, 1.0], [-13.6, -43.0]])
        self.assertTrue(self.sys.check_stability(A_s))
        self.assertFalse(self.sys.check_stability(A_u))


if __name__ == "__main__":
    unittest.main()
