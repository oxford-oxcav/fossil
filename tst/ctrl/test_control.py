from fossil import control
import numpy as np
import sympy as sp
import unittest


class LineariserTests(unittest.TestCase):
    def setUp(self):
        self.model1 = MockModel_NoControl()
        self.model2 = MockModel()
        self.lineariser1 = control.Lineariser(self.model1)
        self.lineariser2 = control.Lineariser(self.model2)

    def test_get_model(self):
        expected_result = [self.lineariser1.x[0] + xi**2 for xi in self.lineariser1.x]
        np.testing.assert_array_equal(self.lineariser1.get_model(), expected_result)
        expected_result = [self.lineariser2.x[0] + xi**2 for xi in self.lineariser2.x]
        np.testing.assert_array_equal(self.lineariser1.get_model(), expected_result)

    def test_get_jacobian(self):
        x0, x1, x2 = self.lineariser1.x
        expected_result = [[2 * x0 + 1, 0, 0], [1, 2 * x1, 0], [1, 0, 2 * x2]]
        np.testing.assert_array_equal(self.lineariser1.get_jacobian(), expected_result)
        x0, x1, x2 = self.lineariser2.x
        expected_result = [[2 * x0 + 1, 0, 0], [1, 2 * x1, 0], [1, 0, 2 * x2]]
        np.testing.assert_array_equal(self.lineariser2.get_jacobian(), expected_result)

    def test_linearise(self):
        x_values = [1.0, 2.0, 3.0]
        expected_result = [[1, 0, 0], [1, 0, 0], [1, 0, 0]]
        np.testing.assert_array_equal(self.lineariser1.linearise(), expected_result)
        np.testing.assert_array_equal(self.lineariser2.linearise(), expected_result)

    def calculate_jacobian(self, model, x_values):
        x_symbols = [sp.Symbol("x" + str(i), real=True) for i in range(len(x_values))]
        J = sp.Matrix(model(x_symbols)).jacobian(x_symbols)
        return J.subs([(x, x_val) for x, x_val in zip(x_symbols, x_values)])


# Mock model class for testing
class MockModel:
    n_vars = 3
    n_u = 1

    def __call__(self, x, u):
        return np.array([x[0] + x[i] ** 2 for i in range(len(x))])


class MockModel_NoControl:
    n_vars = 3

    def __call__(self, x):
        return np.array([x[0] + x[i] ** 2 for i in range(len(x))])


class LQRTest(unittest.TestCase):
    ### Example from https://www.mathworks.com/help/control/ref/lqr.html
    def setUp(self) -> None:
        A = np.array([[0, 1, 0, 0], [0, -0.1, 3, 0], [0, 0, 0, 1], [0, -0.5, 30, 0]])
        B = np.array([[0, 2, 0, 5]]).T
        Q = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
        R = np.array([[1]])
        self.lqr = control.LQR(A, B, Q, R)

    def test_lqr(self):
        expected_result = np.array([[-1, -1.7559, 16.9145, 3.2274]])

        np.testing.assert_array_almost_equal(
            self.lqr.solve(), expected_result, decimal=4
        )


class EigenCalculatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.A = np.array(
            [[0, 1, 0, 0], [0, -0.1, 3, 0], [0, 0, 0, 1], [0, -0.5, 30, 0]]
        )
        self.B = np.array([[0, 2, 0, 5]]).T
        self.Q = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
        self.R = np.array([[1]])

    def test_unstable_ex(self):
        E_u = control.EigenCalculator(self.A)
        self.assertFalse(E_u.is_stable())

    def test_stable_ex(self):
        lqr = control.LQR(self.A, self.B, self.Q, self.R)
        K = lqr.solve()
        E_s = control.EigenCalculator(self.A - self.B @ K)
        self.assertTrue(E_s.is_stable())


if __name__ == "__main__":
    unittest.main()
