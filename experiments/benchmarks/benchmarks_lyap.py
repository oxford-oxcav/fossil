# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
import sympy as sp
import re
import z3 as z3
from experiments.benchmarks.domain_fcns import *
import matplotlib.pyplot as plt


###############################
# NON POLY BENCHMARKS
###############################

# this series comes from
# 2014, Finding Non-Polynomial Positive Invariants and Lyapunov Functions for
# Polynomial Systems through Darboux Polynomials.

# also from CDC 2011, Parrillo, poly system w non-poly lyap
def nonpoly0(batch_size, functions, inner=0.0, outer=10.0):
    _And = functions['And']

    def f(_, v):
        x, y = v
        return [
            -x + x * y,
            -y
        ]

    def XD(_, v):
        x, y = v
        return _And(x > 0, y > 0,
                    inner ** 2 <= x ** 2 + y ** 2, x ** 2 + y ** 2 <= outer**2)

    def SD():
        return slice_init_data((0, 0), outer**2, batch_size)

    return f, XD, SD()


def nonpoly1(batch_size, functions, inner=0.0, outer=10.0):
    _And = functions['And']

    def f(_, v):
        x, y = v
        return  [
                -x + 2*x**2 * y,
                -y
                ]

    def XD(_, v):
        x, y = v
        return _And(x > 0, y > 0, x ** 2 + y ** 2 <= outer**2)

    def SD():
        return slice_init_data((0, 0), outer**2, batch_size)

    return f, XD, SD()


def nonpoly2(batch_size, functions, inner=0.0, outer=10.0):
    _And = functions['And']

    def f(_, v):
        x, y, z = v
        return  [
                -x,
                -2*y + 0.1*x*y**2 + z,
                -z -1.5*y
                ]

    def XD(_, v):
        x, y, z = v
        return _And(x > 0, y > 0, z > 0, x ** 2 + y ** 2 + z**2 <= outer**2)

    def SD():
        return slice_3d_init_data((0, 0, 0), outer**2, batch_size)

    return f, XD, SD()


def nonpoly3(batch_size, functions, inner=0.0, outer=10.0):
    _And = functions['And']

    def f(_, v):
        x, y, z = v
        return  [
                -3*x - 0.1*x*y**3,
                -y + z,
                -z
                ]

    def XD(_, v):
        x, y, z = v
        return _And(x > 0, y > 0, z > 0, x ** 2 + y ** 2 + z**2 <= outer**2)

    def SD():
        return slice_3d_init_data((0, 0, 0), outer**2, batch_size)

    return f, XD, SD()


######################
# POLY benchmarks
######################


def benchmark_0(batch_size, functions, inner=0.0, outer=10.0):
    _And = functions['And']

    # test function, not to be included
    def f(_, v):
        x, y = v
        return [-x, -y]

    def XD(_, v):
        x, y = v
        return _And(x ** 2 + y ** 2 > inner, x ** 2 + y ** 2 <= outer ** 2)

    def SD():
        return circle_init_data((0, 0), outer ** 2, batch_size)

    return f, XD, SD()


def poly_1(batch_size, functions, inner=0.0, outer=10.0):
    # SOSDEMO2
    # from http://sysos.eng.ox.ac.uk/sostools/sostools.pdf
    _And = functions['And']

    # test function, not to be included
    def f(_, v):
        x, y, z = v
        return [
            -x**3 - x*z**2,
            -y - x**2 * y,
            -z + 3*x**2*z - (3*z)
        ]

    def XD(_, v):
        x, y, z = v
        return _And(x ** 2 + y ** 2 + z ** 2 > inner, x ** 2 + y ** 2 + z ** 2 <= outer ** 2)

    def SD():
        return sphere_init_data((0, 0, 0), outer ** 2, batch_size)

    return f, XD, SD()


# this series comes from
# https://www.cs.colorado.edu/~srirams/papers/nolcos13.pdf
# srirams paper from 2013 (old-ish) but plenty of lyap fcns

def poly_2(batch_size, functions, inner=0.0, outer=10.0):
    _And = functions['And']

    def f(_, v):
        # if v.shape[0] == 1:
        #     return [- v[0, 0] ** 3 + v[0, 1], - v[0, 0] - v[0, 1]]
        # else:
        #     return [- v[:, 0] ** 3 + v[:, 1], - v[:, 0] - v[:, 1]]
        x,y = v
        return [- x**3 + y, - x - y]

    def XD(_, v):
        x, y = v
        return _And(x ** 2 + y ** 2 > inner, x ** 2 + y ** 2 <= outer ** 2)

    def SD():
        return circle_init_data((0, 0), outer ** 2, batch_size)

    return f, XD, SD()


def poly_3(batch_size, functions, inner=0.0, outer=10.0):
    _And = functions['And']

    def f(_, v):
        x, y = v
        return [-x**3 - y**2, x*y - y**3]

    def XD(_, v):
        x, y = v
        return _And(x ** 2 + y ** 2 > inner, x ** 2 + y ** 2 <= outer ** 2)

    def SD():
        return circle_init_data((0, 0), outer ** 2, batch_size)

    return f, XD, SD()


def poly_4(batch_size, functions, inner=0.0, outer=10.0):
    _And = functions['And']

    def f(_, v):
        x, y = v
        return [
            -x - 1.5 * x**2 * y**3,
            -y**3 + 0.5 * x**3 * y**2
        ]

    def XD(_, v):
        x, y = v
        return _And(x ** 2 + y ** 2 > inner, x ** 2 + y ** 2 <= outer ** 2)

    def SD():
        return circle_init_data((0, 0), outer ** 2, batch_size)

    return f, XD, SD()


def twod_hybrid(batch_size, functions, inner, outer):
    # example of 2-d hybrid sys
    _And = functions['And']

    def f(functions, v):
        _If = functions['If']
        x0, x1 = v
        _then = - x1 - 0.5*x0**3
        _else = - x1 - x0**2 - 0.25*x1**3
        _cond = x1 >= 0
        return [-x0, _If(_cond, _then, _else)]

    def XD(_, v):
        x0, x1 = v
        return _And(inner**2 < x0**2 + x1**2,
                               x0**2 + x1**2 <= outer**2)

    def SD():
        return circle_init_data((0., 0.), outer**2, batch_size)

    return f, XD, SD()

def linear_discrete(batch_size, functions, inner, outer):
    _And = functions['And']

    def f(_, v):
        x, y = v
        return [0.5 * x - 0.5 * y, 0.5 * x]
    
    def XD(_, v):
        x0, x1 = v
        return _And(inner**2 < x0**2 + x1**2,
                               x0**2 + x1**2 <= outer**2)

    def SD():
        return circle_init_data((0., 0.), outer**2, batch_size)

    return f, XD, SD()


def max_degree_fx(fx):
    return max(max_degree_poly(f) for f in fx)


def max_degree_poly(p):
    s = str(p)
    s = re.sub(r'x\d+', 'x', s)
    try:
        f = sp.sympify(s)
        return sp.degree(f)
    except:
        print("Exception in %s for %s" % (max_degree_poly.__name__, p))
        return 0


if __name__ == '__main__':
    f, XD, SD = poly_1(batch_size=500, functions={'And': z3.And, 'Or': None}, inner=0, outer=10.)
    plt.scatter(SD[:, 0].detach(), SD[:, 1].detach(), color='b')
    plt.show()