import unittest
from unittest import mock
from functools import partial
import numpy as np
from src.lyap.learner.net import NN
from src.shared.Trajectoriser import Trajectoriser
from src.shared.activations import ActivationType
from experiments.benchmarks.benchmarks_lyap import benchmark_3
import torch


class TestLearning(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(90)
        self.n_vars = 2
        system = partial(benchmark_3, batch_size=500)
        self.f, _, self.S_d = system(functions={'And': None})
        self.f_learner = partial(self.f, {'And': None})
        self.Sdot = torch.stack(self.f_learner(self.S_d.T)).T
        self.hidden = [3]
        self.activate = [ActivationType.SQUARE]

    def zero_in_zero(self, learner):
        v_zero, vdot_zero, grad_v = learner.forward_tensors(
            torch.zeros(1, self.n_vars).reshape(1, self.n_vars),
            torch.zeros(self.n_vars, 1).reshape(self.n_vars, 1)
        )
        return v_zero

    def test_lyapunov_conditions(self):
        # lnn = some LNN
        # dynamic = some ODE
        # points = some set of point
        learner = NN(self.n_vars, *self.hidden,
                     bias=False, activate=self.activate, equilibria=None)
        optimizer = torch.optim.AdamW(learner.parameters(), lr=0.1)

        # <training>
        learner.learn(optimizer, self.S_d, self.Sdot, factors=None)

        # self.assertTrue(zero_in_zero(lnn, dynamic))
        v_zero = self.zero_in_zero(learner)
        self.assertEqual(v_zero.item(), 0.0)

        # self.assertTrue(positive_definite(lnn, dynamic))
        v, vdot, _ = learner.forward_tensors(self.S_d, self.Sdot)
        self.assertTrue(all(v >= 0.0))

        # self.assertTrue(negative_lee_derivative(lnn, dynamic))
        self.assertTrue(all(vdot <= 0.0))

    def test_find_closest_unsat(self):
        
        learner = NN(self.n_vars, *self.hidden,
                     bias=False, activate=self.activate, equilibria=None)
        learner.find_closest_unsat(self.S_d, self.Sdot, None)
        self.assertEqual(learner.closest_unsat, torch.tensor(0.008437633514404297))


if __name__ == '__main__':
    unittest.main()
