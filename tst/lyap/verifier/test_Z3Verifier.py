# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from functools import partial
from unittest import mock

from z3 import *

import src.verifier as verifier
import src.learner as learner
from experiments.benchmarks.benchmarks_lyap import *
import src.translator as translator
import src.certificate as certificate
from src.consts import (
    TranslatorType,
    CegisStateKeys,
    ActivationType,
    CegisConfig,
)


class TestZ3Verifier(unittest.TestCase):
    def test_poly2_with_good_Lyapunov_function(self):
        system = poly_2
        n_vars = 2

        ver = verifier.VerifierZ3
        x = ver.new_vars(n_vars)

        f, domain, _, var_bounds = system()
        domain_z3 = domain["lie-&-pos"](x)
        lc = certificate.Lyapunov(domains={"lie-&-pos": domain_z3}, config=CegisConfig)
        ver = verifier.VerifierZ3(n_vars, lc.get_constraints, var_bounds, x, True)

        # model
        model = learner.LearnerCT(2, None, *[2], bias=False)
        with torch.no_grad():
            model.layers[0].weight[0][0] = 1
            model.layers[0].weight[0][1] = 0
            model.layers[0].weight[1][0] = 0
            model.layers[0].weight[1][1] = 1
            model.layers[1].weight[0][0] = 1
            model.layers[1].weight[0][1] = 1

        xdot = f(x)
        tr = translator.TranslatorCT(
            np.array(x).reshape(-1, 1), np.array(xdot).reshape(-1, 1), None, 1, True
        )
        res = tr.get(**{"net": model})
        V, Vdot = res[CegisStateKeys.V], res[CegisStateKeys.V_dot]
        print(V)
        res = ver.verify(V, Vdot)
        self.assertEqual(
            res[CegisStateKeys.found], res[CegisStateKeys.cex] == {"lie-&-pos": []}
        )
        self.assertTrue(res[CegisStateKeys.found])

    def test_poly2_with_bad_Lyapunov_function(self):
        system = poly_2
        n_vars = 2

        ver = verifier.VerifierZ3
        x = ver.new_vars(n_vars)

        f, domain, _, var_bounds = system()
        domain_z3 = domain["lie-&-pos"](x)
        lc = certificate.Lyapunov(domains={"lie-&-pos": domain_z3}, config=CegisConfig)
        ver = verifier.VerifierZ3(n_vars, lc.get_constraints, var_bounds, x, True)

        # model
        model = learner.LearnerCT(2, None, 2, bias=True)

        with torch.no_grad():
            model.layers[0].weight[0][0] = 1
            model.layers[0].weight[0][1] = 0
            model.layers[0].weight[1][0] = 0
            model.layers[0].weight[1][1] = 1
            model.layers[0].bias[0] = 1
            model.layers[0].bias[1] = 1

        xdot = f(x)
        tr = translator.TranslatorCT(
            np.array(x).reshape(-1, 1), np.array(xdot).reshape(-1, 1), None, 1, True
        )
        res = tr.get(**{"net": model})
        V, Vdot = res[CegisStateKeys.V], res[CegisStateKeys.V_dot]
        res = ver.verify(V, Vdot)
        self.assertEqual(res[CegisStateKeys.found], res[CegisStateKeys.cex] == [])
        self.assertFalse(res[CegisStateKeys.found])

    def test_poly2_with_another_bad_Lyapunov_function(self):
        system = poly_2
        n_vars = 2

        ver = verifier.VerifierZ3
        x = ver.new_vars(n_vars)

        f, domain, _, var_bounds = system()
        domain_z3 = domain["lie-&-pos"](x)
        lc = certificate.Lyapunov(domains={"lie-&-pos": domain_z3}, config=CegisConfig)
        ver = verifier.VerifierZ3(n_vars, lc.get_constraints, var_bounds, x, True)

        # model
        model = learner.LearnerCT(2, None, *[2], bias=False)
        with torch.no_grad():
            model.layers[0].weight[0][0] = 1
            model.layers[0].weight[0][1] = 1
            model.layers[0].weight[1][0] = 0
            model.layers[0].weight[1][1] = 1

        xdot = f(x)
        tr = translator.TranslatorCT(
            np.array(x).reshape(-1, 1), np.array(xdot).reshape(-1, 1), None, 1, True
        )
        res = tr.get(**{"net": model})
        V, Vdot = res[CegisStateKeys.V], res[CegisStateKeys.V_dot]
        res = ver.verify(V, Vdot)
        self.assertEqual(res[CegisStateKeys.found], res[CegisStateKeys.cex] == [])
        self.assertFalse(res[CegisStateKeys.found])


if __name__ == "__main__":
    unittest.main()
