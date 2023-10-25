import torch
import torch.nn as nn
import unittest
from unittest.mock import MagicMock
from fossil.learner import LearnerNN, QuadraticFactor, LearnerCT, LearnerDT
from fossil.consts import *


class TestLearnerNN(unittest.TestCase):
    ## Honestly  I got ChatGPT to write these and then corrected them
    def setUp(self):
        self.input_size = 3
        self.learn_method = MagicMock(return_value=None)
        self.nn = LearnerNN(
            self.input_size,
            self.learn_method,
            *[5],
        )

    def test_forward(self):
        x = torch.rand((2, self.input_size))
        y = self.nn(x)
        print(self.nn.layers[-2])
        self.assertEqual(list(y.shape), [2])
        self.assertTrue(
            torch.allclose(
                y.T, self.nn.layers[-1](self.nn.layers[-2](x) ** 2).flatten()
            )
        )

    def test_compute_net_gradnet(self):
        S = torch.rand((2, self.input_size))
        nn, grad_nn = self.nn.compute_net_gradnet(S)
        self.assertEqual(nn.shape, (2,))
        self.assertEqual(grad_nn.shape, (2, self.input_size))

    def test_compute_V_gradV(self):
        nn = torch.ones((2,))
        grad_nn = torch.rand((2, self.input_size))
        S = torch.rand((2, self.input_size))
        V, gradV = self.nn.compute_V_gradV(nn, grad_nn, S)
        self.assertEqual(V.shape, (2,))
        self.assertEqual(gradV.shape, (2, self.input_size))

    def test_compute_factors(self):
        S = torch.rand((2, self.input_size))
        F, derivative_F = self.nn.compute_factors(S)
        self.assertEqual(F, 1.0)
        self.assertEqual(derivative_F, 0.0)
        self.nn.factor = QuadraticFactor()
        F, derivative_F = self.nn.compute_factors(S)
        self.assertEqual(F.shape, (2,))
        self.assertEqual(derivative_F.shape, (2, self.input_size))

    def test_diagonalisation(self):
        self.nn.layers[0].weight = nn.Parameter(torch.rand((3, self.input_size)))
        self.nn.layers[1].weight = nn.Parameter(torch.rand((3, 3)))
        self.nn.diagonalisation()
        self.assertTrue(
            torch.allclose(
                self.nn.layers[0].weight, torch.diag(self.nn.layers[0].weight.diag())
            )
        )


class TestLearnerCT(unittest.TestCase):
    def setUp(self):
        self.input_size = 3
        self.learn_method = MagicMock(return_value=None)
        self.nn = LearnerCT(
            self.input_size,
            self.learn_method,
            *[5],
        )

    def test_get_all(self):
        S = torch.rand((5, self.input_size))
        Sdot = torch.rand((5, self.input_size))
        V, Vdot, circle = self.nn.get_all(S, Sdot)
        self.assertEqual(V.shape, (5,))
        self.assertEqual(Vdot.shape, (5,))
        self.assertEqual(circle.shape, (5,))

    def test_compute_dV(self):
        gradV = torch.randn(2, self.input_size)
        Sdot = torch.randn(2, self.input_size)

        expected_output = torch.sum(torch.mul(gradV, Sdot), dim=1)
        output = self.nn.compute_dV(gradV, Sdot)

        self.assertTrue(torch.allclose(output, expected_output))


class TestLearnerDT(unittest.TestCase):
    def setUp(self):
        self.input_size = 3
        self.learn_method = MagicMock(return_value=None)
        self.nn = LearnerDT(self.input_size, self.learn_method, *[5])

    def test_get_all(self):
        # Define inputs
        S = torch.rand((5, self.input_size))
        Sdot = torch.rand((5, self.input_size))

        # Compute expected outputs
        nn = self.nn.forward(S)
        nn_next = self.nn.forward(Sdot)
        circle = torch.pow(S, 2).sum(dim=1)
        E = self.nn.compute_factors(S)
        V = nn * E[0]
        delta_V = nn_next - V

        # Compute actual outputs
        actual_V, actual_delta_V, actual_circle = self.nn.get_all(S, Sdot)

        # Compare expected and actual outputs
        self.assertTrue(torch.allclose(actual_V, V))
        self.assertTrue(torch.allclose(actual_delta_V, delta_V))
        self.assertTrue(torch.allclose(actual_circle, circle))


if __name__ == "__main__":
    unittest.main()
