import sympy as sp
import re
from experiments.benchmarks.domain_fcns import *


###############################
### NON POLY BENCHMARKS
###############################

# this series comes from
# 2014, Finding Non-Polynomial Positive Invariants and Lyapunov Functions forPolynomial Systems through Darboux Polynomials.

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
        return _And(x ** 2 + y ** 2 > inner**2, x ** 2 + y ** 2 <= outer**2)

    def SD():
        return circle_init_data((0, 0), outer**2, batch_size)

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
        return _And(x ** 2 + y ** 2 > inner, x ** 2 + y ** 2 <= outer**2)

    def SD():
        return circle_init_data((0, 0), outer**2, batch_size)

    return f, XD, SD()


######################
### TACAS benchmarks
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


def benchmark_1(batch_size, functions, inner=0.0, outer=10.0):
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
        x, y = v
        return _And(x ** 2 + y ** 2 > inner, x ** 2 + y ** 2 <= outer ** 2)

    def SD():
        return circle_init_data((0, 0), outer ** 2, batch_size)

    return f, XD, SD()


# this series comes from
# https://www.cs.colorado.edu/~srirams/papers/nolcos13.pdf
# srirams paper from 2013 (old-ish) but plenty of lyap fcns

def benchmark_3(batch_size, functions, inner=0.0, outer=10.0):
    _And = functions['And']

    def f(_, v):
        # x,y = v
        if v.shape[0] == 1:
            return [- v[0, 0] ** 3 + v[0, 1], - v[0, 0] - v[0, 1]]
        else:
            return [- v[:, 0] ** 3 + v[:, 1], - v[:, 0] - v[:, 1]]
        # return [- x**3 + y, - x - y]

    def XD(_, v):
        x, y = v
        return _And(x ** 2 + y ** 2 > inner, x ** 2 + y ** 2 <= outer ** 2)

    def SD():
        return circle_init_data((0, 0), outer ** 2, batch_size)

    return f, XD, SD()


def benchmark_4(batch_size, functions, inner=0.0, outer=10.0):
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


def benchmark_5(batch_size, functions, inner=0.0, outer=10.0):
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


def benchmark_6(batch_size, functions, inner=0.0, outer=10.0):
    _And = functions['And']

    def f(_, v):
        x, y, w, z = v
        return [-x + y**3 - 3*w*z, -x - y**3, x*z - w, x*w - z**3]

    def XD(_, v):
        x, y = v
        return _And(x ** 2 + y ** 2 > inner, x ** 2 + y ** 2 <= outer ** 2)

    def SD():
        return circle_init_data((0, 0), outer ** 2, batch_size)

    return f, XD, SD()


def benchmark_7(batch_size, functions, inner=0.0, outer=10.0):
    _And = functions['And']

    def f(_, v):
        x0, x1, x2, x3, x4, x5 = v
        return [
            - x0 ** 3 + 4 * x1 ** 3 - 6 * x2 * x3,
            -x0 - x1 + x4 ** 3,
            x0 * x3 - x2 + x3 * x5,
            x0 * x2 + x2 * x5 - x3 ** 3,
            - 2 * x1 ** 3 - x4 + x5,
            -3 * x2 * x3 - x4 ** 3 - x5
        ]

    def XD(_, v):
        x, y = v
        return _And(x ** 2 + y ** 2 > inner, x ** 2 + y ** 2 <= outer ** 2)

    def SD():
        return circle_init_data((0, 0), outer ** 2, batch_size)

    return f, XD, SD()


def benchmark_8(x):
    # todo: parametric model
    return [
        x[1],
        -(m+2)*x[0] - x[1]
    ]

def benchmark_9(x):
    # todo: parametric model
    return [
        x[1],
        -(m+2)*x[0] - x[1]
    ]


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
