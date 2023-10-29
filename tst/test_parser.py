# Copyright (c) 2023, Alessandro Abate, Alec Edwards, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable

import unittest

import dreal
import z3
from cvc5 import pythonic as cvpy
import sympy
import numpy as np

from fossil.parser import (
    parse_expression,
    parse_domain,
    parse_dynamical_system_to_numpy,
    SymbolicParsingError,
    DomainParsingError,
)
from fossil import domains


def compare_without_whitespace(str1, str2):
    return str1.replace(" ", "").replace("\t", "").replace("\n", "") == str2.replace(
        " ", ""
    ).replace("\t", "").replace("\n", "")


class TestParserZ3(unittest.TestCase):
    def test_z3_integer(self):
        s = "-5"
        result = parse_expression(s)
        self.assertEqual(result, -5)

    def test_z3_decimal(self):
        s = "1.5"
        result = parse_expression(s)
        self.assertEqual(result, 1.5)

    def test_z3_variable(self):
        s = "x3"
        result = parse_expression(s)
        self.assertIsInstance(result, z3.ArithRef)
        self.assertTrue(compare_without_whitespace(str(result), s))

    def test_z3_negative_variable(self):
        s = "-x3"
        result = parse_expression(s)
        self.assertIsInstance(result, z3.ArithRef)
        self.assertTrue(compare_without_whitespace(str(result), s))

    def test_z3_arithmetic(self):
        s = "x0 + x1 * 2 - 3 / x2"
        result = parse_expression(s)
        self.assertIsInstance(result, z3.ArithRef)
        self.assertTrue(compare_without_whitespace(str(result), s))

    def test_z3_complex(self):
        s = "1.5*x1+x2+ x3*x2*x1**2"
        s_compare = "3/2*x1+x2+x3*x2*x1**2"
        result = parse_expression(s, output="z3")
        self.assertIsInstance(result, z3.ArithRef)
        self.assertTrue(compare_without_whitespace(str(result), s_compare))

    def test_z3_control(self):
        s = "1.5*x1+2*x2+ x3*x2*x1**2 + u0"
        s_compare = "3/2*x1+2*x2+ x3*x2*x1**2 + u0"
        result = parse_expression(s, output="z3")
        self.assertIsInstance(result, z3.ArithRef)
        self.assertTrue(compare_without_whitespace(str(result), s_compare))

    # def test_z3_function(self):
    #     s = "If(x0, x1, x2)"
    #     result = parse_expression(s)
    #     self.assertIsInstance(result, z3.BoolRef)

    def test_invalid_expression(self):
        s = "invalid_expr"
        with self.assertRaises(SymbolicParsingError):
            parse_expression(s)

    def test_invalid_output_format(self):
        s = "x0"
        with self.assertRaises(ValueError):
            parse_expression(s, output="invalid_output")


class TestParserCVC(unittest.TestCase):
    def test_cvpy_integer(self):
        s = "-5"
        result = parse_expression(s, output="cvc5")
        self.assertEqual(result, -5)

    def test_cvpy_decimal(self):
        s = "1.5"
        result = parse_expression(s, output="cvc5")
        self.assertEqual(result, 1.5)

    def test_cvpy_variable(self):
        s = "x3"
        result = parse_expression(s, output="cvc5")
        self.assertIsInstance(result, cvpy.ArithRef)
        self.assertTrue(compare_without_whitespace(str(result), s))

    def test_cvpy_negative_variable(self):
        s = "-x3"
        result = parse_expression(s, output="cvc5")
        self.assertIsInstance(result, cvpy.ArithRef)
        self.assertTrue(compare_without_whitespace(str(result), s))

    def test_cvpy_arithmetic(self):
        s = "x0 + x1 * 2 - 3 / x2"
        result = parse_expression(s, output="cvc5")
        self.assertIsInstance(result, cvpy.ArithRef)
        self.assertTrue(compare_without_whitespace(str(result), s))

    def test_cvpy_complex(self):
        s = "1.5*x1+x2+ x3*x2*x1**2"
        s_compare = "3/2*x1+x2+x3*x2*x1**2"
        result = parse_expression(s, output="cvc5")
        self.assertIsInstance(result, cvpy.ArithRef)
        self.assertTrue(compare_without_whitespace(str(result), s_compare))

    def test_cvpy_control(self):
        s = "1.5*x1+2*x2+ x3*x2*x1**2 + u0"
        s_compare = "3/2*x1+2*x2+ x3*x2*x1**2 + u0"
        result = parse_expression(s, output="cvc5")
        self.assertIsInstance(result, cvpy.ArithRef)
        self.assertTrue(compare_without_whitespace(str(result), s_compare))

    # def test_cvpy_function(self):
    #     s = "If(x0, x1, x2)"
    #     result = parse_expression(s, output="cvc5")
    #     self.assertIsInstance(result, cvpy.BoolRef)

    def test_invalid_expression(self):
        s = "invalid_expr"
        with self.assertRaises(SymbolicParsingError):
            parse_expression(s, output="cvc5")

    def test_invalid_output_format(self):
        s = "x0"
        with self.assertRaises(ValueError):
            parse_expression(s, output="invalid_output")


# Sympy and Dreal rearrange expressions internally, so we can't compare the strings


class TestParserDreal(unittest.TestCase):
    def test_dreal_integer(self):
        s = "5"
        result = parse_expression(s, output="dreal")
        self.assertEqual(result, 5)

    def test_dreal_decimal(self):
        s = "1.5"
        result = parse_expression(s, output="dreal")
        self.assertEqual(result, 1.5)

    def test_dreal_variable(self):
        s = "x3"
        result = parse_expression(s, output="dreal")
        self.assertIsInstance(result, dreal.Expression)

    def test_dreal_negative_variable(self):
        s = "-x3"
        result = parse_expression(s, output="dreal")
        self.assertIsInstance(result, dreal.Expression)

    def test_dreal_arithmetic(self):
        s = "x0 + x1 * 2 - 3 / x2"
        result = parse_expression(s, output="dreal")
        self.assertIsInstance(result, dreal.Expression)

    def test_dreal_complex(self):
        s = "1.5*x1+x2+ sin(x3*x2*x1)"
        result = parse_expression(s, output="dreal")
        self.assertIsInstance(result, dreal.Expression)

    def test_dreal_control(self):
        s = "1.5*x1+x2+ sin(x3*x2*x1) + u0"
        result = parse_expression(s, output="dreal")
        self.assertIsInstance(result, dreal.Expression)

    def test_dreal_function(self):
        s = "sin(x1)"
        result = parse_expression(s, output="dreal")
        self.assertIsInstance(result, dreal.Expression)


class TestParserSympy(unittest.TestCase):
    def test_sympy_integer(self):
        s = "-5"
        result = parse_expression(s, output="sympy")
        self.assertEqual(result, -5)

    def test_sympy_decimal(self):
        s = "1.5"
        result = parse_expression(s, output="sympy")
        self.assertEqual(result, 1.5)

    def test_sympy_variable(self):
        s = "x3"
        result = parse_expression(s, output="sympy")
        self.assertIsInstance(result, sympy.Symbol)
        self.assertTrue(compare_without_whitespace(str(result), s))

    def test_sympy_negative_variable(self):
        s = "-x3"
        result = parse_expression(s, output="sympy")
        self.assertIsInstance(
            result, sympy.Mul
        )  # Since -x3 is a multiplication of -1 and x3
        self.assertTrue(compare_without_whitespace(str(result), s))

    def test_sympy_arithmetic(self):
        s = "x0 + 2 * x1  - 3 / x2"
        result = parse_expression(s, output="sympy")
        self.assertIsInstance(
            result, sympy.Add
        )  # Since this is an arithmetic expression

    def test_sympy_complex(self):
        s = "1.5*x1+x2+ x3*x2*x1**2"
        s_compare = "3/2*x1+x2+x3*x2*x1**2"
        result = parse_expression(s, output="sympy")
        self.assertIsInstance(result, sympy.Add)

    def test_sympy_function(self):
        s = "sin(x1)"
        result = parse_expression(s, output="sympy")
        self.assertIsInstance(result, sympy.sin)
        self.assertTrue(compare_without_whitespace(str(result), s))


class TestDomainsParser(unittest.TestCase):
    def testSphere(self):
        s = "Sphere([3.0, 0], 3)"
        result = parse_domain(s)
        self.assertIsInstance(result, domains.Sphere)
        self.assertEqual(result.radius, 3)
        self.assertEqual(result.centre, [3.0, 0])

    def testRectangle(self):
        s = "Rectangle([0, 1.0], [2.0, 3.0])"
        result = parse_domain(s)
        self.assertIsInstance(result, domains.Rectangle)
        self.assertEqual(result.lower_bounds, [0, 1.0])
        self.assertEqual(result.upper_bounds, [2.0, 3.0])

    def testTorus(self):
        s = "Torus([0, 1.0], 2.0, 1.0)"
        result = parse_domain(s)
        self.assertIsInstance(result, domains.Torus)
        self.assertEqual(result.centre, [0, 1.0])
        self.assertEqual(result.inner_radius, 1.0)
        self.assertEqual(result.outer_radius, 2.0)

    def testInvalidDomain(self):
        s = "InvalidDomain([0, 1.0], 1.0, 2.0)"
        with self.assertRaises(DomainParsingError):
            parse_domain(s)


class TestParseDynamicalSystemToNumpy(unittest.TestCase):
    def test_no_controls(self):
        dynamical_system = ["x0 + 2*x1", "x1 - 3*x0"]
        funcs = parse_dynamical_system_to_numpy(dynamical_system)
        self.assertEqual(len(funcs), 2)

        # Evaluate the parsed functions at x0=1, x1=2
        result1 = funcs[0](np.array([1, 2]))
        result2 = funcs[1](np.array([1, 2]))

        self.assertAlmostEqual(result1, 5)
        self.assertAlmostEqual(result2, -1)

    def test_with_controls(self):
        # This test is currently broken because the parser interprets the final expression as
        # "x1 - 3*x0 - u1". I'm not sure why atm.
        dynamical_system = ["x0 + 2*x1 + u0", "x1 - (3*x0) + u1"]
        funcs = parse_dynamical_system_to_numpy(dynamical_system)
        self.assertEqual(len(funcs), 2)

        # Evaluate the parsed functions at x0=1, x1=2 and u0=1, u1=-1
        result1 = funcs[0](np.array([1, 2]), np.array([1, -1]))
        result2 = funcs[1](np.array([1, 2]), np.array([1, -1]))

        self.assertAlmostEqual(result1, 6)
        self.assertAlmostEqual(result2, -2)

    def test_mixed_symbols(self):
        dynamical_system = ["x1**2 + u0", "x0*x1 + u1"]
        funcs = parse_dynamical_system_to_numpy(dynamical_system)
        self.assertEqual(len(funcs), 2)

        # Evaluate the parsed functions at x0=2, x1=3 and u0=1, u1=-1
        result1 = funcs[0](np.array([2, 3]), np.array([1, -1]))
        result2 = funcs[1](np.array([2, 3]), np.array([1, -1]))

        self.assertAlmostEqual(result1, 10)  # 3^2 + 1
        self.assertAlmostEqual(result2, 5)  # 2*3 - 1


if __name__ == "__main__":
    unittest.main()
