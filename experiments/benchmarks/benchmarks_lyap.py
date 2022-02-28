# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
from typing import Any

import sympy as sp
import re
from matplotlib import pyplot as plt

from experiments.benchmarks.domain_fcns import *
import experiments.benchmarks.models as models


###############################
# NON POLY BENCHMARKS
###############################

# this series comes from
# 2014, Finding Non-Polynomial Positive Invariants and Lyapunov Functions for
# Polynomial Systems through Darboux Polynomials.

# also from CDC 2011, Parrillo, poly system w non-poly lyap

def nonpoly0_lyap():
    p = models.NonPoly0()
    domain = Torus([0,0], 10, 0.1)

    return p, {'lie-&-pos': domain.generate_domain}, {'lie-&-pos':domain.generate_data(1000)}, inf_bounds_n(2)

def nonpoly0_rws():
    p = models.NonPoly0()
    XD = Sphere([0,0], 10)
    goal = Sphere([0,0], 0.1)
    unsafe = Sphere([3,3], 0.5)
    init = Sphere([-3, -3], 0.5)
    batch_size = 500
    domains = {'lie': XD.generate_domain,
                'init': init.generate_domain,
                'unsafe': unsafe.generate_boundary,
                'goal':goal.generate_domain}

    data = {'lie': SetMinus(XD, goal).generate_data(batch_size), 
            'init': init.generate_data(batch_size),
            'unsafe': unsafe.generate_data(batch_size)}

    return p, domains, data, inf_bounds_n(2)

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

    return f, [XD], [SD()], inf_bounds_n(2)


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

    return f, [XD], [SD()], inf_bounds_n(3)


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

    return f, [XD], [SD()], inf_bounds_n(3)


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

    return f, [XD], [SD()], inf_bounds_n(2)


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

    return f, [XD], [SD()], inf_bounds_n(3)


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

    return f, [XD], [SD()], inf_bounds_n(2)


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

    return f, [XD], [SD()], inf_bounds_n(2)


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

    return f, [XD], [SD()], inf_bounds_n(2)

 
def twod_hybrid():
    # example of 2-d hybrid sys

    f = models.Hybrid2d()
    batch_size = 1000

    XD = Torus([0,0], 10, 0.00001)

    return f, {'lie-&-pos':XD.generate_domain}, {'lie-&-pos': XD.generate_data(batch_size)}, inf_bounds_n(2)

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

    return f, [XD], [SD()], inf_bounds_n(2)


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


# if __name__ == '__main__':
#     f, [XD], SD, _ = poly_1(batch_size=500, functions={'And': z3.And, 'Or': None}, inner=0, outer=10.)
#     plt.scatter(SD[0][:, 0].detach(), SD[0][:, 1].detach(), color='b')
#     plt.show()