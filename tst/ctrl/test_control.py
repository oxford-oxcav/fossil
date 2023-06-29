from src import control
import numpy as np
import sympy as sp
import unittest


class LineariserTests(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()  # Replace with your actual model implementation
        self.lineariser = control.Lineariser(self.model)

    def test_get_model(self):
        expected_result = [self.lineariser.x[0] + xi**2 for xi in self.lineariser.x]
        np.testing.assert_array_equal(self.lineariser.get_model(), expected_result)

    def test_get_jacobian(self):
        x0, x1, x2 = self.lineariser.x
        expected_result = [[2 * x0 + 1, 0, 0], [1, 2 * x1, 0], [1, 0, 2 * x2]]
        np.testing.assert_array_equal(self.lineariser.get_jacobian(), expected_result)

    def test_linearise(self):
        x_values = [1.0, 2.0, 3.0]
        expected_result = [[1, 0, 0], [1, 0, 0], [1, 0, 0]]
        np.testing.assert_array_equal(self.lineariser.linearise(), expected_result)

    def calculate_jacobian(self, model, x_values):
        x_symbols = [sp.Symbol("x" + str(i), real=True) for i in range(len(x_values))]
        J = sp.Matrix(model(x_symbols)).jacobian(x_symbols)
        return J.subs([(x, x_val) for x, x_val in zip(x_symbols, x_values)])


# Mock model class for testing
class MockModel:
    n_vars = 3

    def __call__(self, x):
        return np.array([x[0] + x[i] ** 2 for i in range(len(x))])


class LQRTest(unittest.TestCase):
    ### Example from https://www.mathworks.com/help/control/ref/lqr.html
    def setUp(self) -> None:
        A = np.array([[0, 1, 0, 0], [0, -0.1, 3, 0], [0, 0, 0, 1], [0, -0.5, 30, 0]])
        print(A)
        B = np.array([[0, 2, 0, 5]]).T
        print(B)
        Q = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
        print(Q)
        R = np.array([[1]])
        self.lqr = control.LQR(A, B, Q, R)

    def test_lqr(self):
        expected_result = np.array([[-1, -1.7559, 16.9145, 3.2274]])

        np.testing.assert_array_almost_equal(
            self.lqr.solve(), expected_result, decimal=4
        )


if __name__ == "__main__":
    unittest.main()
