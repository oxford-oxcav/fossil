# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sympy as sp
from z3 import ArithRef, simplify

from fossil.primer import Primer
from fossil.activations import ActivationType
from fossil.consts import *
from fossil.shared.cegis_values import CegisConfig
from fossil.domains import Rectangle, Sphere

exp, sin, cos = sp.exp, sp.sin, sp.cos


def simplify_f(function):
    """
    Attempts to simplify symbolic function:
    :param function: symbolic function of type sympy.exp, z3.ArithRef or dreal
    :returns f: simplified function
    """
    if isinstance(function, sp.exp):
        f = sp.simplify(function)
        return f
    if isinstance(function, ArithRef):
        f = simplify(function)
        return f
    else:
        return function


def print_f(function):
    """
    Attempts to simplify and print symbolic function:
    :param function: symbolic function of type sympy.exp, z3.ArithRef or dreal
    """
    print(simplify_f(function))


def initialise_states(N):
    """
    :param N: int, number of states to initialise
    :return states: tuple of states as symp vars x0,...,xN
    """
    states = " ".join(["x%d" % i for i in range(N)])
    v = sp.symbols(states, real=True)
    return v


def synthesise(f, mode, **kw):
    """
    Main synthesis function.
    :param f: vector field dynamics f as list of symbolic expressions.
    :param mode: PrimerMode, lyapunov or barrier synthesis
    :returns C_n, C_s: Numerical and symbolic versions of synthesised certificate
    """
    p = Primer.create_Primer(f, mode, **kw)
    return p.get()
