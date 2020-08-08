from __future__ import division
from z3 import *
import sympy as sp

try:
    import dreal as dr
except:
    print('No dreal')

# adapted from https://stackoverflow.com/questions/22488553/how-to-use-z3py-and-sympy-together
from barrier.drealverifier import DRealVerifier
from barrier.z3verifier import Z3Verifier


def _sympy_converter(var_map, exp, target, expand_pow=False):
    rv = None
    assert isinstance(exp, sp.Expr) and target is not None

    if isinstance(exp, sp.Symbol):
        rv = var_map.get(exp.name, None)
    elif isinstance(exp, sp.Number):
        try:
            rv = RealVal(exp) if isinstance(target, Z3Verifier) else sp.RealNumber(exp)
        except:  # Z3 parser error
            rep = sp.Float(exp, len(str(exp)))
            rv = RealVal(rep)
    elif isinstance(exp, sp.Add):
        # Add(exp_0, ...)
        rv = _sympy_converter(var_map, exp.args[0], target, expand_pow=expand_pow)  # eval this expression
        for e in exp.args[1:]:  # add it to all other remaining expressions
            rv += _sympy_converter(var_map, e, target, expand_pow=expand_pow)
    elif isinstance(exp, sp.Mul):
        rv = _sympy_converter(var_map, exp.args[0], target, expand_pow=expand_pow)
        for e in exp.args[1:]:
            rv *= _sympy_converter(var_map, e, target, expand_pow=expand_pow)
    elif isinstance(exp, sp.Pow):
        x = _sympy_converter(var_map, exp.args[0], target, expand_pow=expand_pow)
        e = _sympy_converter(var_map, exp.args[1], target, expand_pow=expand_pow)
        if expand_pow:
            try:
                i = float(e.sexpr())
                assert i.is_integer()
                i = int(i) - 1
                rv = x
                for _ in range(i):
                    rv *= x
            except:  # fallback
                rv = _sympy_converter(var_map, exp, target, expand_pow=False)
        else:
            rv = x ** e
    elif isinstance(exp, sp.Function):
        # check various activation types ONLY FOR DREAL
        if isinstance(exp, sp.tanh):
            rv = dr.tanh(_sympy_converter(var_map, exp.args[0], target, expand_pow=expand_pow))
        elif isinstance(exp, sp.sin):
            rv = dr.sin(_sympy_converter(var_map, exp.args[0], target, expand_pow=expand_pow))
        elif isinstance(exp, sp.cos):
            rv = dr.cos(_sympy_converter(var_map, exp.args[0], target, expand_pow=expand_pow))
        elif isinstance(exp, sp.exp):
            rv = dr.exp(_sympy_converter(var_map, exp.args[0], target, expand_pow=expand_pow))
        else:
            ValueError('Term ' + str(exp) + ' not recognised')

    assert rv is not None
    return rv


def sympy_converter(exp, target=Z3Verifier, var_map={}):
    return _sympy_converter(var_map, exp, target)
