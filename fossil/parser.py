# Copyright (c) 2023, Alessandro Abate, Alec Edwards, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable

from functools import partial

import z3
import sympy as sp
import dreal
from cvc5 import pythonic as cvpy
import pyparsing as pp
import torch

from fossil import domains
from fossil import logger

parser_log = logger.Logger.setup_logger(__name__)


class SymbolicParsingError(Exception):
    """Exception raised when there is an error parsing a symbolic expression"""

    pass


class DomainParsingError(Exception):
    """Exception raised when there is an error parsing a domain"""

    pass


def get_var_by_index(t):
    """From a token, return the variable number assuming variable names are of the form x0,x1,..."""
    return t[0][1:]


def func_names_to_match_first(funcs):
    return pp.MatchFirst(pp.Keyword(fn) for fn in funcs.keys())


class SymbolicParser:
    """Base class for parsing symbolic expressions."""

    def __init__(self):
        self.xs = {}
        self.us = {}
        self.funcs = self.get_funcs()
        self.variable = self.get_variable()
        self.expr = self.create_grammar()

    def parse_dynamical_system(self, s) -> list:
        """Parse a  list of strings into a list of symbolic expressions

        This function ensures that when each string is parsed, the smt variables
        of the whole dynamical system are collected as part of this object.
        Args:
            s (str): A string representing a symbolic expression

        Returns:
            A list symbolic expression
        """

        dynamical_system = []
        for expr in s:
            parsed = self.expr.parseString(expr, parseAll=True).asList()
            symbolic_expr = self.convert_parse_to_ast(parsed)
            dynamical_system.append(symbolic_expr)

        return dynamical_system

    def get_funcs(self):
        return {}

    def create_grammar(self):
        """Create the grammar for the parser"""
        decimal = pp.Combine(pp.Word(pp.nums) + "." + pp.Word(pp.nums)).setParseAction(
            lambda t: float(t[0])
        )
        integer = pp.Word(pp.nums).setParseAction(lambda t: int(t[0]))
        number = decimal | integer

        unary_minus = pp.Literal("-").setParseAction(lambda: "u-")

        variable = self.variable

        func_names = func_names_to_match_first(self.funcs)
        expr = pp.Forward()

        # Parentheses for grouping in arithmetic and for function arguments
        nested_paren_expr = pp.nestedExpr("(", ")", content=expr)

        # Logic function call
        func_call = pp.Group(func_names + nested_paren_expr)

        expr <<= pp.infixNotation(
            number | variable | func_call | nested_paren_expr,
            [
                (unary_minus, 1, pp.opAssoc.RIGHT),
                ("**", 2, pp.opAssoc.RIGHT),
                ("/", 2, pp.opAssoc.LEFT),
                ("*", 2, pp.opAssoc.LEFT),
                ("-", 2, pp.opAssoc.LEFT),
                ("+", 2, pp.opAssoc.LEFT),
                # subtraction before addition because it seems to fix a bug
            ],
        )
        return expr

    def convert_parse_to_ast(self, parsed):
        """Convert a parsed expression to an AST.

        Args:
            parsed (list): A parsed expression.

        Returns:
            An abstract syntax tree (AST).
        """
        funcs = self.funcs
        binary_ops = {
            "+": lambda lhs, rhs: lhs + rhs,
            "-": lambda lhs, rhs: lhs - rhs,
            "*": lambda lhs, rhs: lhs * rhs,
            "/": lambda lhs, rhs: lhs / rhs,
            "**": lambda lhs, rhs: lhs**rhs,
        }
        unary_ops = {
            "u-": lambda operand: -operand,
        }

        if isinstance(parsed, list):
            if len(parsed) == 1:
                return self.convert_parse_to_ast(parsed[0])
            elif len(parsed) >= 3 and len(parsed) % 2 == 1:
                op = parsed[1]
                op_func = binary_ops[op]

                # Determine operands based on length
                operands = parsed[::2] if len(parsed) > 3 else parsed[0::2]

                args = [self.convert_parse_to_ast(x) for x in operands]
                result = args[0]
                for arg in args[1:]:
                    result = op_func(result, arg)
                return result
            elif parsed[0] in unary_ops:
                operand = self.convert_parse_to_ast(parsed[1])
                return unary_ops[parsed[0]](operand)
            elif parsed[0] in funcs:
                func, arity = funcs[parsed[0]]
                args = [self.convert_parse_to_ast(arg) for arg in parsed[1]]
                return func(*args)
            else:
                # Handle the case when there's a nested structure
                # Not sure if this is needed
                flattened = []
                for p in parsed:
                    if isinstance(p, list):
                        flattened.extend(p)
                    else:
                        flattened.append(p)
                return self.convert_parse_to_ast(flattened)

        return parsed

    def get_variable(self):
        """Create the grammar for state and control variables"""
        variable_x = pp.Word("x", pp.nums).setParseAction(self.make_var)
        variable_u = pp.Word("u", pp.nums).setParseAction(self.make_var)

        return variable_x | variable_u

    def make_var(self, t):
        """Create a variable from a parsed token

        Args:
            t : A parsed token

        Raises:
            ValueError: Varaibles must start with x or u

        Returns:
            symbolic variable: A symbolic variable
        """
        var_name = t[0]
        if var_name in self.xs:
            return self.xs[var_name]
        var = self.var_function(var_name)
        if var_name.startswith("x"):
            self.xs[var_name] = var
        elif var_name.startswith("u"):
            self.us[var_name] = var
        else:
            # Hopefully this never happens
            raise ValueError(f"Invalid variable name {var_name}")
        return var

    @staticmethod
    def var_function(name: str):
        """Abstract method for generating symbolic variables based on the given name.

        Args:
            name (str): Variable name.

        Returns:
            Symbolic variable.

        Raises:
            NotImplementedError: This method needs to be implemented by derived classes.
        """
        raise NotImplementedError


class Z3Parser(SymbolicParser):
    """Parser for Z3 expressions"""

    def get_funcs(self):
        z3_funcs = {
            "Not": (z3.Not, 1),
            "And": (z3.And, 2),
            "Or": (z3.Or, 2),
            "If": (z3.If, 3),
            "sum": (z3.Sum, 1),
        }
        return z3_funcs

    @staticmethod
    def var_function(name: str):
        parser_log.debug(f"Creating z3 variable {name}")
        return z3.Real(name)

    def subsitution_function(self, expr, xs):
        # xs is a list of the form [0, x1, ...}]from cegis
        # expr is a z3 expression
        subs_dict = [(self.xs[str(variable)], variable) for variable in xs]
        res = z3.substitute(expr, *subs_dict)
        return res

    def substition_function_u(self, expr, xs, ux):
        # xs is a list of the form [0, x1, ...}]from cegis
        # expr is a dreal expression (with control variables, u0, ...)
        # ux is a list of stack feedback equations (e.g. 1.5*x1 + x2)
        # we need to substitute these in for u0,... in expr
        sorted_us = sorted(self.us.items(), key=lambda x: x[0])
        subs_dict = [(self.xs[str(variable)], variable) for variable in xs]
        subs_dict.extend([(u, ux[i]) for i, (_, u) in enumerate(sorted_us)])
        res = z3.substitute(expr, *subs_dict)
        return res

    def parse_dynamical_system(self, s):
        ds = super().parse_dynamical_system(s)
        if len(self.us) > 0:
            return [partial(self.substition_function_u, expr) for expr in ds]
        else:
            return [partial(self.subsitution_function, expr) for expr in ds]


class CVC5Parser(SymbolicParser):
    """Parser for CVC5 expressions"""

    def get_funcs(self):
        cvpy_funcs = {
            "Not": (cvpy.Not, 1),
            "And": (cvpy.And, 2),
            "Or": (cvpy.Or, 2),
            "If": (cvpy.If, 3),
            "sum": (cvpy.Sum, 1),
        }
        return cvpy_funcs

    @staticmethod
    def var_function(name: str):
        parser_log.debug(f"Creating z3 variable {name}")
        return cvpy.Real(name)

    def subsitution_function(self, expr, xs):
        # xs is a list of the form [0, x1, ...}]from cegis
        # expr is a z3 expression
        subs_dict = [(self.xs[str(variable)], variable) for variable in xs]
        res = cvpy.substitute(expr, *subs_dict)
        return res

    def substition_function_u(self, expr, xs, ux):
        # xs is a list of the form [0, x1, ...}]from cegis
        # expr is a dreal expression (with control variables, u0, ...)
        # ux is a list of stack feedback equations (e.g. 1.5*x1 + x2)
        # we need to substitute these in for u0,... in expr
        sorted_us = sorted(self.us.items(), key=lambda x: x[0])
        subs_dict = [(self.xs[str(variable)], variable) for variable in xs]
        subs_dict.extend([(u, ux[i]) for i, (_, u) in enumerate(sorted_us)])
        res = cvpy.substitute(expr, *subs_dict)
        return res

    def parse_dynamical_system(self, s):
        ds = super().parse_dynamical_system(s)
        if len(self.us) > 0:
            return [partial(self.substition_function_u, expr) for expr in ds]
        else:
            return [partial(self.subsitution_function, expr) for expr in ds]


class DrealParser(SymbolicParser):
    """Parser for dReal expressions"""

    def get_funcs(self):
        dreal_funcs = {
            "Not": (dreal.Not, 1),
            "And": (dreal.And, 2),
            "Or": (dreal.Or, 2),
            "sin": (dreal.sin, 1),
            "cos": (dreal.cos, 1),
            "exp": (dreal.exp, 1),
            "log": (dreal.log, 1),
            "sqrt": (dreal.sqrt, 1),
            "tanh": (dreal.tanh, 1),
            "sum": (sum, 1),
        }
        return dreal_funcs

    @staticmethod
    def var_function(name: str):
        #  We must store the variable in a dictionary, because it is later used as a key and keys must be hashable
        # We must also return a dreal expression so it can be substituted
        parser_log.debug(f"Creating dreal variable {name}")
        return dreal.Variable(name)

    def make_var(self, t):
        """Create a variable from a parsed token

        Args:
            t : A parsed token

        Raises:
            ValueError: Varaibles must start with x or u

        Returns:
            symbolic variable: A symbolic variable
        """

        var_name = t[0]
        if var_name in self.xs:
            return self.xs[var_name]
        var = self.var_function(var_name)
        if var_name.startswith("x"):
            self.xs[var_name] = var
        elif var_name.startswith("u"):
            self.us[var_name] = var
        else:
            # Hopefully this never happens
            raise ValueError(f"Invalid variable name {var_name}")
        return 1 * var

    def subsitution_function(self, expr, xs):
        # xs is a list of the form [0, x1, ...}]from cegis
        # expr is a dreal expression
        subs_dict = {self.xs[str(variable)]: variable for variable in xs}
        return expr.Substitute(subs_dict)

    def substition_function_u(self, expr, xs, ux):
        # xs is a list of the form [0, x1, ...}]from cegis
        # expr is a dreal expression (with control variables, u0, ...)
        # ux is a list of stack feedback equations (e.g. 1.5*x1 + x2)
        # we need to substitute these in for u0,... in expr
        sorted_us = sorted(self.us.items(), key=lambda x: x[0])
        subs_dict = {self.xs[str(variable)]: variable for variable in xs}
        subs_dict.update({u: ux[i] for i, (_, u) in enumerate(sorted_us)})
        return expr.Substitute(subs_dict)

    def parse_dynamical_system(self, s) -> list[callable]:
        ds = super().parse_dynamical_system(s)
        if len(self.us) > 0:
            return [partial(self.substition_function_u, expr) for expr in ds]
        else:
            return [partial(self.subsitution_function, expr) for expr in ds]


class SympyParser(SymbolicParser):
    """Parser for sympy expressions"""

    def get_funcs(self):
        sympy_funcs = {
            "Not": (sp.Not, 1),
            "And": (sp.And, 2),
            "Or": (sp.Or, 2),
            "If": (sp.ITE, 3),
            "sum": (sum, 1),
            "sin": (sp.sin, 1),
            "cos": (sp.cos, 1),
            "exp": (sp.exp, 1),
            "log": (sp.log, 1),
            "sqrt": (sp.sqrt, 1),
            "tanh": (sp.tanh, 1),
        }
        return sympy_funcs

    @staticmethod
    def var_function(name: str):
        parser_log.debug(f"Creating sympy variable {name}")
        return sp.var(name)


class DomainParser:
    """Parser for domains"""

    def __init__(self):
        self.expr = self.create_grammar()

    def create_grammar(self):
        """Create the grammar for the domain parser"""
        decimal = pp.Combine(
            pp.Optional(pp.oneOf("+ -")) + pp.Word(pp.nums) + "." + pp.Word(pp.nums)
        ).setParseAction(lambda t: float(t[0]))
        integer = pp.Combine(
            pp.Optional(pp.oneOf("+ -")) + pp.Word(pp.nums)
        ).setParseAction(lambda t: int(t[0]))
        number = decimal | integer
        number_list = pp.nestedExpr("[", "]", content=pp.delimitedList(number))
        sphere_input = pp.Group(number_list + pp.Suppress(",") + number)
        sphere = pp.Keyword("Sphere") + pp.nestedExpr("(", ")", content=sphere_input)
        rectangle_input = pp.Group(number_list + pp.Suppress(",") + number_list)
        rectangle = pp.Keyword("Rectangle") + pp.nestedExpr(
            "(", ")", content=rectangle_input
        )
        torus_input = pp.Group(
            number_list + pp.Suppress(",") + number + pp.Suppress(",") + number
        )
        torus = pp.Keyword("Torus") + pp.nestedExpr("(", ")", content=torus_input)
        domain = sphere | rectangle | torus

        sphere.setParseAction(self.make_sphere)
        rectangle.setParseAction(self.make_rectangle)
        torus.setParseAction(self.make_torus)

        domain_expr = pp.Forward()
        # Domain parser. TODO: Maybe add domain operations like intersection, union, etc.
        domain_expr <<= domain
        return domain_expr

    @staticmethod
    def make_sphere(t):
        """Make a sphere object from a parsed sphere token"""
        center = t.asList()[1][0][0]
        radius = t.asList()[1][0][1]
        return domains.Sphere(center, radius)

    @staticmethod
    def make_rectangle(t):
        """Make a rectangle object from a parsed rectangle token"""
        lb = t.asList()[1][0][0]
        ub = t.asList()[1][0][1]
        return domains.Rectangle(lb, ub)

    @staticmethod
    def make_torus(t):
        """Make a torus object from a parsed torus token"""
        center = t.asList()[1][0][0]
        out_radius = t.asList()[1][0][1]
        inner_radius = t.asList()[1][0][2]
        return domains.Torus(center, out_radius, inner_radius)


def __lambdify_to_numpy(expr, state_symbols):
    """Convert a sympy expression to a numpy function

    This may only be called on expressions parsed using this module to
    ensure the input is sanitized.

    Args:
        expr (sympy.Expr): A sympy expression
        state_symbols (list): A list of sympy symbols that are state variables, in order

    Returns:
            A numpy function
    """
    torch_func = {
        "sin": torch.sin,
        "cos": torch.cos,
        "exp": torch.exp,
    }
    return sp.lambdify([state_symbols], expr, modules=[torch_func, "numpy"])


def _lamdify_to_numpy_u(expr, state_symbols, control_symbols):
    """Convert a sympy expression to a numpy function

    This may only be called on expressions parsed using this module to
    ensure the input is sanitized.

    Args:
        expr (sympy.Expr): A sympy expression
        state_symbols (list): A list of sympy symbols that are state variables, in order
        control_symbols (list): A list of sympy symbols that are control variables, in order
    Returns:
            A numpy function
    """
    torch_func = {
        "sin": torch.sin,
        "cos": torch.cos,
        "exp": torch.exp,
    }
    return sp.lambdify(
        [state_symbols, control_symbols], expr, modules=[torch_func, "numpy"]
    )


def parse_expression(s, output="z3"):
    """Parse a string into a symbolic expression

    Args:
        s (str): A string representing a symbolic expression
        output (str, optional): The output format. Defaults to "z3".
    """
    if output == "z3":
        parser = Z3Parser()
    elif output == "cvc5":
        parser = CVC5Parser()
    elif output == "dreal":
        parser = DrealParser()
    elif output == "sympy":
        parser = SympyParser()
    else:
        raise ValueError("Invalid output format")

    try:
        parsed = parser.expr.parseString(s, parseAll=True).asList()
        symbolic_expr = parser.convert_parse_to_ast(parsed)
        return symbolic_expr
    except pp.ParseException as e:
        raise SymbolicParsingError(f"Error parsing expression {s}") from e


def parse_domain(s):
    """Parse a string into a domain object (e.g. Sphere, Rectangle, Torus)

    Args:
        s (str): A string representing a domain

    Returns:
        A domain object
    """
    try:
        domain_parser = DomainParser()
        domain_expr = domain_parser.expr
        return domain_expr.parseString(s, parseAll=True).asList()[0]
    except pp.ParseException as e:
        raise DomainParsingError(f"Error parsing domain {s}") from e


def parse_dynamical_system_to_numpy(dynamical_system: list[str]):
    """Parse a list of strings into a numpy function

    This function ensures that when each string is parsed, the free variables
    of the whole dynamical system are collected. This ensures that the
    resulting function takes inputs correctly.

    Args:
        s (str): A list of string string representing a vector valued function

    Returns:
        A numpy function
    """
    p = SympyParser()
    exprs = p.parse_dynamical_system(dynamical_system)
    # exprs = [parse_expression(s, output="sympy") for s in dynamical_system]
    free_symbols = set()
    # TODO: check here that u0 is present if u1 or more is present?
    # Are similar checks needed?
    for expr in exprs:
        free_symbols.update(expr.free_symbols)
    x = [s for s in free_symbols if s.name.startswith("x")]
    u = [s for s in free_symbols if s.name.startswith("u")]

    # Sort the symbols to ascending order (assume they are named x0, x1, x2, ...)
    x.sort(key=lambda s: s.name)
    u.sort(key=lambda s: s.name)

    if len(u) > 0:
        return [_lamdify_to_numpy_u(expr, x, u) for expr in exprs]
    else:
        return [__lambdify_to_numpy(expr, x) for expr in exprs]


if __name__ == "__main__":
    s = "1.5*x1+x2 + sin(x3*x2*x1*x4)"
    result = parse_expression(s, output="dreal")
    print(result)
    s2 = "x1 - 3*x0 + u1"
    result2 = parse_expression(s2, output="dreal")
    print(result2)
    s3 = "x1*x2*x3*x4*x5"
    result3 = parse_expression(s3, output="dreal")
    print(result3)
    s4 = "-x1"
    result4 = parse_expression(s4, output="dreal")
    print(result4)
