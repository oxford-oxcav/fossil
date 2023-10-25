# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from z3 import *
import sympy as sp

import logging

from fossil.sympy_converter import sympy_converter

try:
    import dreal as dr

    with_dreal = True
except ImportError as e:
    with_dreal = False
    logging.exception("Could not import dreal")


class UtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)
        self.x_sp, self.y_sp = sp.symbols("x y")
        self.f = -3 * sp.sin(self.x_sp) - 4 * self.x_sp + 5 * self.y_sp**2

    def test_whenSympyConverterCalledWithZ3_thenReturnsZ3Expression(self):
        syms = {
            "x": Real("x"),
            "y": Real("y"),
            "sin": lambda x: x,
        }

        converted = sympy_converter(syms, self.f, to_number=lambda x: RealVal(x))
        expected = -3 * syms["x"] - 4 * syms["x"] + 5 * syms["y"] ** 2

        s = Solver()
        s.add(expected == converted)
        self.assertTrue(s.check() == sat)

    def test_whenSympyConverterCalledWithDReal_thenReturnsDRealExpression(self):
        if not with_dreal:
            return
        syms = {
            "x": dr.Variable("x"),
            "y": dr.Variable("y"),
            "sin": dr.sin,
            "cos": dr.cos,
            "sqrt": dr.sqrt,
            "pow": dr.pow,
            "log": dr.log,
            "acos": dr.acos,
            "asin": dr.asin,
            "atan": dr.atan,
            "atan2": dr.atan2,
            "cosh": dr.cosh,
            "sinh": dr.sinh,
            "tanh": dr.tanh,
        }
        converted = sympy_converter(syms, self.f)
        expected = -3 * dr.sin(syms["x"]) - 4 * syms["x"] + 5 * syms["y"] ** 2
        fml = converted == expected
        self.assertEqual(dr.CheckSatisfiability(fml, 0.0000001).size(), 0)


if __name__ == "__main__":
    unittest.main()
