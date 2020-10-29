from unittest import mock

import unittest

from z3 import Reals

from src.lyap.verifier.verifier import Verifier


class SimplifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.z3_vars = Reals('x y z')

    def test_whenGetSimplifiableVdot_returnSimplifiedVdot(self):
        # inputs
        x, y, z = self.z3_vars
        f = x * y + 2 * z
        domain = x*x + y*y + z*z <= 1
        return_value = 'result'
        t = 1

        with mock.patch.object(Verifier, '_solver_solve') as s:
            # setup
            s.return_value = return_value
            v = Verifier(3, 0, domain, self.z3_vars)
            v.timeout = t

            # call tested function
            res, timedout = v.solve_with_timeout(None, f)

        # assert results
        self.assertEqual(res, return_value)
        self.assertFalse(timedout)


if __name__ == '__main__':
    unittest.main()
