# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import unittest

from z3 import Reals

import fossil.verifier as verifier
import fossil.certificate as certificate
from fossil.consts import CegisConfig


class SimplifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.z3_vars = Reals("x y z")

    def test_whenGetSimplifiableVdot_returnSimplifiedVdot(self):
        # inputs
        x, y, z = self.z3_vars
        f = x * y + 2 * z
        domain = x * x + y * y + z * z <= 1
        return_value = "result"
        t = 1
        lc = certificate.Lyapunov(domains={"lie": domain}, config=CegisConfig())

        with mock.patch.object(verifier.Verifier, "_solver_solve") as s:
            # setup
            s.return_value = return_value
            v = verifier.Verifier(3, lc.get_constraints, self.z3_vars, True)
            v.timeout = t

            # call tested function
            res, timedout = v.solve_with_timeout(None, f)

        # assert results
        self.assertEqual(res, return_value)
        self.assertFalse(timedout)


if __name__ == "__main__":
    unittest.main()
