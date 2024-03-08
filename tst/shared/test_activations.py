import unittest

import dreal
import numpy as np
import sympy as sp
import torch

from fossil import activations as activations
from fossil import consts


class TestActivations(unittest.TestCase):
    def test_derivatives(self):
        x = torch.rand((100, 10), requires_grad=True)
        for act in consts.ActivationType:
            if act in (consts.ActivationType.RATIONAL,):
                continue
            act = activations.activation_fcn(act)
            sig = act(x)
            derivative = torch.autograd.grad(
                outputs=sig,
                inputs=x,
                grad_outputs=torch.ones_like(sig),
                create_graph=True,
                retain_graph=True,
            )[0]
            # self.assertTrue(torch.allclose(sig_der, true_der))
            self.assertIsNotNone(sig)
            self.assertIsNotNone(derivative)


class TestSymbolicActivations(unittest.TestCase):
    def test_derivatives(self):
        x = [dreal.Variable("x" + str(i)) for i in range(1, 11)]
        coeffs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        p = np.atleast_2d([ci * xi for xi, ci in zip(x, coeffs)]).T
        for act in consts.ActivationType:
            if act in (
                consts.ActivationType.RELU,
                consts.ActivationType.RELU_SQUARE,
                consts.ActivationType.REQU,
                consts.ActivationType.RATIONAL,
            ):
                # We could test these but we never need their derivatives anyway
                continue
            print(act)
            act_sym = activations.activation_fcn(act)
            sig = act_sym.forward_symbolic(p)
            sig_der = act_sym.derivative_symbolic(p)
            self.assertIsNotNone(sig)
            self.assertIsNotNone(sig_der)
            for xi, ci, sig_i, sig_der_i in zip(x, coeffs, sig, sig_der):
                true_der = sig_i.item().Differentiate(xi)
                # Derivative functions exclude factor of ci
                der = sig_der_i.item() * ci
                diff = sp.simplify(str(true_der - der))
                self.assertEqual(diff, 0)

    def test_numerically_similar(self):
        x = [dreal.Variable("x" + str(i)) for i in range(1, 11)]
        p = np.atleast_2d([1 * xi for xi in x]).T
        data = torch.rand((5, 10))
        for act in consts.ActivationType:
            if act in (consts.ActivationType.RATIONAL,):
                # We could test these but we never need their derivatives anyway
                continue
            print(act)
            act = activations.activation_fcn(act)
            sig = act(data)
            sig_sym = act.forward_symbolic(p)
            for data_i, sig_i in zip(data, sig):
                sig_i = sig_i.detach().numpy()
                for xi, data_ij, sym_i, zi in zip(x, data_i, sig_sym, sig_i):
                    sym_i = sym_i.item().Evaluate({xi: data_ij.item()})
                    self.assertTrue(np.isclose(zi, sym_i, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
