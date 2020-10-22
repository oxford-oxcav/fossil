import unittest
from src.lyap.verifier.z3verifier import Z3Verifier
from functools import partial
from src.lyap.learner.net import NN
from src.shared.activations import ActivationType
from experiments.benchmarks.benchmarks_lyap import *
import torch
from src.shared.Regulariser import Regulariser 
from unittest import mock
from z3 import *

class TestZ3Verifier(unittest.TestCase):

    def test_benchmark3_with_good_Lyapunov_function(self):
        system = partial(benchmark_3, batch_size=100)
        n_vars = 2

        verifier = Z3Verifier
        x = verifier.new_vars(n_vars)

        f, domain, _ = system(functions=verifier.solver_fncts(), inner=0, outer=100)
        domain_z3 = domain(verifier.solver_fncts(), x)
        verifier = Z3Verifier(n_vars, f, domain_z3, x)

        # model
        model = NN(2, 2,
                   bias=False,
                   activate=[ActivationType.SQUARE],
                   equilibria=None)
        model.layers[0].weight[0][0] = 1
        model.layers[0].weight[0][1] = 0
        model.layers[0].weight[1][0] = 0
        model.layers[0].weight[1][1] = 1
        model.layers[1].weight[0][0] = 1
        model.layers[1].weight[0][1] = 1
        
        xdot = f(Z3Verifier.solver_fncts(), x)
        regulariser = Regulariser(model, np.matrix(x).T, xdot, None, 1)
        res = regulariser.get(**{'factors': None})
        V, Vdot = res['V'], res['V_dot']
        print(V)
        res = verifier.verify(V, Vdot)
        self.assertEqual(res['found'], res['cex'] == [])
        self.assertTrue(res['found'])

    def test_benchmark3_with_bad_Lyapunov_function(self):
        system = partial(benchmark_3, batch_size=100)
        n_vars = 2

        verifier = Z3Verifier
        x = verifier.new_vars(n_vars)

        f, domain, _ = system(functions=verifier.solver_fncts(), inner=0, outer=100)
        domain_z3 = domain(verifier.solver_fncts(), x)
        verifier = Z3Verifier(n_vars, f, domain_z3, x)

        # model
        model = NN(2, 2,
                   bias=True,
                   activate=[ActivationType.SQUARE],
                   equilibria=None)
        model.layers[0].weight[0][0] = 1
        model.layers[0].weight[0][1] = 0
        model.layers[0].weight[1][0] = 0
        model.layers[0].weight[1][1] = 1
        
        model.layers[0].bias[0] = 1
        model.layers[0].bias[1] = 1
        
        xdot = f(Z3Verifier.solver_fncts(), x)
        regulariser = Regulariser(model, np.matrix(x).T, xdot, None, 1)
        res = regulariser.get(**{'factors': None})
        V, Vdot = res['V'], res['V_dot']
        res = verifier.verify(V, Vdot)
        self.assertEqual(res['found'], res['cex'] == [])
        self.assertFalse(res['found'])

    def test_benchmark3_with_another_bad_Lyapunov_function(self):
        system = partial(benchmark_3, batch_size=100)
        n_vars = 2

        verifier = Z3Verifier
        x = verifier.new_vars(n_vars)

        f, domain, _ = system(functions=verifier.solver_fncts(), inner=0, outer=100)
        domain_z3 = domain(verifier.solver_fncts(), x)
        verifier = Z3Verifier(n_vars, f, domain_z3, x)

        # model
        model = NN(2, 2,
                   bias=False,
                   activate=[ActivationType.SQUARE],
                   equilibria=None)
        model.layers[0].weight[0][0] = 1
        model.layers[0].weight[0][1] = 1
        model.layers[0].weight[1][0] = 0
        model.layers[0].weight[1][1] = 1
        
        xdot = f(Z3Verifier.solver_fncts(), x)
        regulariser = Regulariser(model, np.matrix(x).T, xdot, None, 1)
        res = regulariser.get(**{'factors': None})
        V, Vdot = res['V'], res['V_dot']
        res = verifier.verify(V, Vdot)
        self.assertEqual(res['found'], res['cex'] == [])
        self.assertFalse(res['found'])
        
if __name__ == '__main__':
    unittest.main()
