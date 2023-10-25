# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import division
from z3 import *
import sympy as sp

try:
    import dreal as dr
except:
    print("No dreal")

# adapted from https://stackoverflow.com/questions/22488553/how-to-use-z3py-and-sympy-together
import fossil.verifier as verifier


def sympy_converter(
    syms: {}, exp: sp.Expr, to_number=lambda x: float(x), expand_pow=True
):
    rv = None
    assert isinstance(exp, sp.Expr)
    if isinstance(exp, sp.Symbol):
        rv = syms.get(exp.name, None)
    elif isinstance(exp, sp.Number):
        try:
            rv = to_number(exp)
        except:  # Z3 parser error?
            rep = sp.Float(exp, len(str(exp)))
            rv = RealVal(rep)
    elif isinstance(exp, sp.Add):
        # Add(exp_0, ...)
        rv = sympy_converter(
            syms, exp.args[0], to_number, expand_pow=expand_pow
        )  # eval this expression
        for e in exp.args[1:]:  # add it to all other remaining expressions
            rv += sympy_converter(syms, e, to_number, expand_pow=expand_pow)
    elif isinstance(exp, sp.Mul):
        rv = sympy_converter(syms, exp.args[0], to_number, expand_pow=expand_pow)
        for e in exp.args[1:]:
            rv *= sympy_converter(syms, e, to_number, expand_pow=expand_pow)
    elif isinstance(exp, sp.Pow):
        x = sympy_converter(syms, exp.args[0], to_number, expand_pow=expand_pow)
        e = sympy_converter(syms, exp.args[1], to_number, expand_pow=expand_pow)
        if expand_pow:
            try:
                i = float(e.sexpr())
                assert i.is_integer()
                i = int(i) - 1
                rv = x
                for _ in range(i):
                    rv *= x
            except:  # fallback
                rv = sympy_converter(syms, exp, to_number, expand_pow=False)
        elif "pow" in syms:
            rv = syms["pow"](x, e)
        else:
            rv = x**e
    elif isinstance(exp, sp.Function):
        for f in [sp.tanh, sp.sin, sp.cos, sp.exp]:
            if isinstance(exp, f):
                a = sympy_converter(syms, exp.args[0], to_number, expand_pow=expand_pow)
                rv = syms.get(f.__name__)(a)
                break
    else:
        ValueError("Term " + str(exp) + " not recognised")
    if rv is None:
        raise ValueError(
            "Could not convert exp:{} (type:{}) with syms:{}".format(
                exp, type(exp), syms
            )
        )
    return rv
