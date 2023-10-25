# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sympy as sp
from fossil.domains import *
from experiments.reactive_modules.ReactiveModule import Atom


def create_benchmark_for_lyap(dynamics, batch_size, functions, inner=0.0, outer=10.0):
    """
    :param dynamics: list of symbolic exprs
    :param batch_size:
    :param functions:
    :param inner: float
    :param outer: float
    :return:
    """
    _And = functions["And"]
    _If = functions["If"]

    def f(_, v):
        if isinstance(dynamics, Atom):
            sp_vars = dynamics.get_vars()
            guards = dynamics.get_guards()
            # assume mutually exclusive guards (*very* easy case)
            dyna = []
            dyna += [sp.lambdify(sp_vars, d) for d in dynamics.get_dynamics()]
            # return if(guard, f1, f2)
            _cond = sp.lambdify(sp_vars, guards[0])(*v)[0] <= 0
            _then = dyna[0](*v)
            _else = dyna[1](*v)
            return [
                _If(_cond, _then[0][0], _else[0][0]),
                _If(_cond, _then[1][0], _else[1][0]),
            ]
        else:
            if len(dynamics) == 1:
                return dynamics(*v)
            else:
                ValueError("not implemented")

    def XD(_, v):
        # x, y = v
        sum_squared_vars = np.sum([x**2 for x in v])
        # x ** 2 > inner is ok
        # x ** 2 >= inner *NO* : x can be == inner --> if inner=0, we ask V(0) > 0
        return _And(sum_squared_vars > inner, sum_squared_vars <= outer**2)

    # needed sth to deduct dimensions
    def SD(v):
        return round_init_data([0 for i in v], outer**2, batch_size)

    return f, XD, SD


def create_benchmark_for_barrier(dynamics, batch_size, functions):
    _And = functions["And"]

    print("Need to set domains")

    def f(_, v):
        return dynamics(*v)

    # todo: get user input, compatibility for And/Or
    def XD(_, v):
        return _And(-3.5 <= x, x <= 2, -2 <= y, y <= 1)

    def XI(_, v):
        x, y = v
        return _And((x - 1.5) ** 2 + y**2 <= 0.25)

    def XU(_, v):
        x, y = v
        return (x + 1) ** 2 + (y + 1) ** 2 <= 0.16

    # d = [ [x_low, y_low], [x_up, y_up] ]
    d = [[-3.5, -2], [2, 1]]

    def SD():
        return square_init_data(d, batch_size)

    def SI():
        return circle_init_data((1.5, 0), 0.25, batch_size)

    def SU():
        return circle_init_data((-1, -1), 0.16, batch_size)

    return f, XD, XI, XU, SD(), SI(), SU(), inf_bounds_n(2)
