# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
import unittest
from functools import partial
from unittest import mock

from z3 import *

from src.verifier.z3verifier import Z3Verifier
from src.learner.net_continuous import NNContinuous
from src.shared.activations import ActivationType
from experiments.benchmarks.benchmarks_lyap import *
from src.translator.translator_continuous import TranslatorContinuous
from src.certificate.lyapunov_certificate import LyapunovCertificate
from src.shared.cegis_values import CegisStateKeys
from src.shared.consts import TranslatorType


class TestZ3Verifier(unittest.TestCase):

    def test_poly2_with_good_Lyapunov_function(self):
        system = partial(poly_2, batch_size=100)
        n_vars = 2

        verifier = Z3Verifier
        x = verifier.new_vars(n_vars)

        f, domain, _, var_bounds = system(functions=verifier.solver_fncts(), inner=0, outer=100)
        domain_z3 = domain[0](verifier.solver_fncts(), x)
        lc = LyapunovCertificate(XD=domain_z3)
        verifier = Z3Verifier(n_vars, lc.get_constraints, domain_z3, var_bounds, x)

        # model
        model = NNContinuous(2, None, 2,
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
        translator = TranslatorContinuous(model, np.array(x).reshape(-1, 1), np.array(xdot).reshape(-1,1), None, 1)
        res = translator.get(**{'factors': None})
        V, Vdot = res[CegisStateKeys.V], res[CegisStateKeys.V_dot]
        print(V)
        res = verifier.verify(V, Vdot)
        self.assertEqual(res[CegisStateKeys.found], res[CegisStateKeys.cex] == [[]])
        self.assertTrue(res[CegisStateKeys.found])

    def test_poly2_with_bad_Lyapunov_function(self):
        system = partial(poly_2, batch_size=100)
        n_vars = 2

        verifier = Z3Verifier
        x = verifier.new_vars(n_vars)

        f, domain, _, var_bounds = system(functions=verifier.solver_fncts(), inner=0, outer=100)
        domain_z3 = domain[0](verifier.solver_fncts(), x)
        lc = LyapunovCertificate(XD=domain_z3)
        verifier = Z3Verifier(n_vars, lc.get_constraints, domain_z3, var_bounds, x)

        # model
        model = NNContinuous(2, None, 2,
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
        translator = TranslatorContinuous(model, np.array(x).reshape(-1, 1), np.array(xdot).reshape(-1,1), None, 1)
        res = translator.get(**{'factors': None})
        V, Vdot = res[CegisStateKeys.V], res[CegisStateKeys.V_dot]
        res = verifier.verify(V, Vdot)
        self.assertEqual(res[CegisStateKeys.found], res[CegisStateKeys.cex] == [])
        self.assertFalse(res[CegisStateKeys.found])

    def test_poly2_with_another_bad_Lyapunov_function(self):
        system = partial(poly_2, batch_size=100)
        n_vars = 2

        verifier = Z3Verifier
        x = verifier.new_vars(n_vars)

        f, domain, _, var_bounds = system(functions=verifier.solver_fncts(), inner=0, outer=100)
        domain_z3 = domain[0](verifier.solver_fncts(), x)
        lc = LyapunovCertificate(XD=domain_z3)
        verifier = Z3Verifier(n_vars, lc.get_constraints, domain_z3, var_bounds, x)

        # model
        model = NNContinuous(2, None, 2,
                   bias=False,
                   activate=[ActivationType.SQUARE],
                   equilibria=None)
        model.layers[0].weight[0][0] = 1
        model.layers[0].weight[0][1] = 1
        model.layers[0].weight[1][0] = 0
        model.layers[0].weight[1][1] = 1
        
        xdot = f(Z3Verifier.solver_fncts(), x)
        translator = TranslatorContinuous(model, np.array(x).reshape(-1, 1), np.array(xdot).reshape(-1,1), None, 1)
        res = translator.get(**{'factors': None})
        V, Vdot = res[CegisStateKeys.V], res[CegisStateKeys.V_dot]
        res = verifier.verify(V, Vdot)
        self.assertEqual(res[CegisStateKeys.found], res[CegisStateKeys.cex] == [])
        self.assertFalse(res[CegisStateKeys.found])


if __name__ == '__main__':
    unittest.main()
