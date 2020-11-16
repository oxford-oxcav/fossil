import math

import matplotlib.pyplot as plt
from experiments.benchmarks.domain_fcns import *

# this series comes from
# Synthesizing Barrier Certificates Using Neural Networks
# by Zhao H. et al
# HSCC 2020
from z3 import z3

inf = 1e300
inf_bounds = [-inf, inf]


def inf_bounds_n(n):
    return [inf_bounds] * n


def prajna07_simple(batch_size, functions):
    _And = functions['And']

    def f(_, v):
        x, y = v
        return [y, - x - y + 1 / 3 * x ** 3]

    def XD(_, v):
        x, y = v
        return _And(-3.5 <= x, x <= 2, -2 <= y, y <= 1)

    def XI(_, v):
        x, y = v
        return _And((x - 1.5) ** 2 + y ** 2 <= 0.25)

    def XU(_, v):
        x, y = v
        return (x + 1)**2 + (y + 1)**2 <= 0.16

    # d = [ [x_low, y_low], [x_up, y_up] ]
    d = [[-3.5, -2], [2, 1]]

    def SD():
        return square_init_data(d, batch_size)

    def SI():
        return circle_init_data((1.5, 0), 0.25, batch_size)

    def SU():
        return circle_init_data((-1, -1), 0.16, batch_size)

    return f, XD, XI, XU, SD(), SI(), SU(), inf_bounds_n(2)


def barr_3(batch_size, functions):
    _And = functions['And']
    _Or = functions['Or']

    def f(_, v):
        x, y = v
        return [y, - x - y + 1.0/3.0 * x ** 3]

    def XD(_, v):
        x, y = v
        return _And(-3 <= x, x <= 2.5, -2 <= y, y <= 1)

    def XI(_, v):
        x, y = v
        return _Or(
            _And((x - 1.5)**2 + y**2 <= 0.25),
            _And(x >= -1.8, x <= -1.2, y >= -0.1, y <= 0.1),
            _And(x >= -1.4, x <= -1.2, y >= -0.5, y <= 0.1),
        )

    def XU(_, v):
        x, y = v
        return _Or(
            (x + 1)**2 + (y + 1)**2 <= 0.16,
            _And(0.4 <= x, x <= 0.6, 0.1 <= y, y <= 0.5),
            _And(0.4 <= x, x <= 0.8, 0.1 <= y, y <= 0.3),
        )

    def SD():
        return square_init_data([[-3.5, -2.5], [3, 1.5]], batch_size)

    epsilon = 0

    def SI():
        n0 = int(batch_size / 3)
        n1 = n0
        n2 = batch_size - (n0 + n1)
        return torch.cat([
            circle_init_data((1.5, 0.), 0.25+epsilon, n0),
            square_init_data([[-1.8, -0.1], [-1.2, 0.1]], n1),
            add_corners_2d([[-1.8, -0.1], [-1.2, 0.1]]),
            square_init_data([[-1.4, -0.5], [-1.2, 0.1]], n2),
            add_corners_2d([[-1.4, -0.5], [-1.2, 0.1]])
        ])

    def SU():
        n0 = int(batch_size / 3)
        n1 = n0
        n2 = batch_size - (n0 + n1)
        return torch.cat([
            circle_init_data((-1., -1.), 0.16+epsilon, n0),
            square_init_data([[0.4, 0.1], [0.6, 0.5]], n1),
            add_corners_2d([[0.4, 0.1], [0.6, 0.5]]),
            square_init_data([[0.4, 0.1], [0.8, 0.3]], n2),
            add_corners_2d([[0.4, 0.1], [0.8, 0.3]])
        ])

    return f, XD, XI, XU, SD(), SI(), SU(), inf_bounds_n(2)


def barr_1(batch_size, functions):
    _And = functions['And']

    def f(_, v):
        x, y = v
        return [y + 2*x*y, -x - y**2 + 2*x**2]

    def XD(_, v):
        x, y = v
        return _And(-2 <= x, x <= 2, -2 <= y, y <= 2)
    def XI(_, v):
        x, y = v
        return _And(0 <= x, x <= 1, 1 <= y, y <= 2)

    def XU(_, v):
        x, y = v
        return x + y**2 <= 0

    def SD():
        domain = [[-2, -2], [2, 2]]
        dom = square_init_data(domain, batch_size)
        return dom

    def SI():
        # domain = [ [x_l, y_l], [x_u, y_u] ]
        domain = [ [0, 1], [1, 2] ]
        dom = square_init_data(domain, batch_size)
        return dom

    def SU():
        # find points in parabola x + y**2 <= 0
        points = []
        limits = [[-2, -2], [0, 2]]
        while len(points) < batch_size:
            dom = square_init_data(limits, batch_size)
            idx = torch.nonzero(dom[:, 0] + dom[:, 1]**2 <= 0)
            points += dom[idx][:, 0, :]
        return torch.stack(points[:batch_size])

    return f, XD, XI, XU, SD(), SI(), SU(), inf_bounds_n(2)


def barr_2(batch_size, functions):
    _And = functions['And']

    def f(functions, v):
        x, y = v
        return [functions['exp'](-x) + y - 1, -(functions['sin'](x)) ** 2]

    def XD(_, v):
        x, y = v
        return _And(-2 <= x, y <= 2)

    def XI(_, v):
        x, y = v
        return (x+0.5)**2 + (y-0.5)**2 <= 0.16

    def XU(_, v):
        x, y = v
        return (x-0.7)**2 + (y+0.7)**2 <= 0.09

    def SD():
        x_comp = -2 + torch.randn(batch_size, 1)**2
        y_comp = 2 - torch.randn(batch_size, 1)**2
        dom = torch.cat([x_comp, y_comp], dim=1)
        return dom

    def SI():
        return circle_init_data((-0.5, 0.5), 0.16, batch_size)

    def SU():
        return circle_init_data((0.7, -0.7), 0.09, batch_size)

    return f, XD, XI, XU, SD(), SI(), SU(), inf_bounds_n(2)


def twod_hybrid(batch_size, functions):
    # A = [ [0,1,0], [0,0,1] [-0.2,-0.3,-1] ]
    # B = [0, 0, 0.1]
    # C = [1, 0, 0]  --> output is x0
    # input : if y >= 0: u=10, else: u=-10
    # X: x0**2 + x1**2 + x2**2 <= 16
    # Xi (x0+2)**2 + x1**2 + x2**2 <= 0.01
    # Xi (x0-2)**2 + x1**2 + x2**2 <= 0.01

    def f(functions, v):
        _If = functions['If']
        x0, x1 = v
        _then = - x0 - 0.5*x0**3
        _else = x0 - 0.25*x1**2
        _cond = x0 >= 0
        return [x1, _If(_cond, _then, _else)]

    def XD(_, v):
        x0, x1 = v
        return x0**2 + x1**2 <= 4

    def XI(_, v):
        x0, x1 = v
        return (x0+1)**2 + (x1+1)**2 <= 0.25

    def XU(_, v):
        x0, x1 = v
        return (x0-1)**2 + (x1-1)**2 <= 0.25

    def SD():
        return circle_init_data((0., 0.), 4, batch_size)

    def SI():
        return circle_init_data((-1., -1.), 0.25, batch_size)

    def SU():
        return circle_init_data((1., 1.), 0.25, batch_size)

    return f, XD, XI, XU, SD(), SI(), SU(), inf_bounds_n(2)


# 4-d ODE benchmark
def hi_ord_4(batch_size, functions):
    _And = functions['And']
    _Or = functions['Or']

    def f(_, v):
        x0, x1, x2, x3 = v
        # x^4 + 3980 x^3 + 4180 x^2 + 2400 x + 576
        # is stable with complex roots
        return [x1, x2,
                x3,
                - 3980*x3 - 4180*x2 - 2400*x1 - 576*x0
                ]

    def XD(_, v):
        x0, x1, x2, x3 = v
        return _And( x0 ** 2 + x1 ** 2 + x2 ** 2 + x3 ** 2 <= 4)

    def XI(_, v):
        x0, x1, x2, x3 = v
        return (x0 - 1.0) ** 2 + (x1 - 1.0) ** 2 + \
               (x2 - 1.0) ** 2 + (x3 - 1.0) ** 2 <= 0.25

    def XU(_, v):
        x0, x1, x2, x3 = v
        return (x0 + 2) ** 2 + (x1 + 2) ** 2 + \
               (x2 + 2) ** 2 + (x3 + 2) ** 2 <= 0.16

    def SD():
        return n_dim_sphere_init_data([0, 0, 0, 0], 4, batch_size)

    epsilon = 0

    def SI():
        return n_dim_sphere_init_data([1.0, 1.0, 1.0, 1.0], 0.25, batch_size)

    def SU():
        return n_dim_sphere_init_data([-2., -2., -2., -2.], 0.16, batch_size)

    return f, XD, XI, XU, SD(), SI(), SU(), inf_bounds_n(4)


# 6-d ODE benchmark
def hi_ord_6(batch_size, functions):
    _And = functions['And']
    _Or = functions['Or']

    def f(_, v):
        x0, x1, x2, x3, x4, x5 = v
        # x^6 + 800 x^5 + 2273 x^4 + 3980 x^3 + 4180 x^2 + 2400 x + 576
        # is stable with complex roots
        return [x1, x2,
                x3, x4,
                x5,
                - 800*x5 - 2273*x4 - 3980*x3 - 4180*x2 - 2400*x1 - 576*x0
                ]

    def XD(_, v):
        x0, x1, x2, x3, x4, x5 = v
        return _And( x0 ** 2 + x1 ** 2 + x2 ** 2 + x3 ** 2 + x4 ** 2 + x5 ** 2 <= 4)

    def XI(_, v):
        x0, x1, x2, x3, x4, x5 = v
        return (x0 - 1.0) ** 2 + (x1 - 1.0) ** 2 + \
               (x2 - 1.0) ** 2 + (x3 - 1.0) ** 2 + \
               (x4 - 1.0) ** 2 + (x5 - 1.0) ** 2 <= 0.25

    def XU(_, v):
        x0, x1, x2, x3, x4, x5 = v
        return (x0 + 2) ** 2 + (x1 + 2) ** 2 + \
               (x2 + 2) ** 2 + (x3 + 2) ** 2 + \
               (x4 + 2) ** 2 + (x5 + 2) ** 2 <= 0.16

    def SD():
        return n_dim_sphere_init_data([0, 0, 0, 0, 0, 0], 4, batch_size)

    epsilon = 0

    def SI():
        return n_dim_sphere_init_data([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.25, batch_size)

    def SU():
        return n_dim_sphere_init_data([-2., -2., -2., -2., -2., -2.], 0.16, batch_size)

    return f, XD, XI, XU, SD(), SI(), SU(), inf_bounds_n(6)


# 8-d ODE benchmark
def hi_ord_8(batch_size, functions):
    _And = functions['And']
    _Or = functions['Or']

    def f(_, v):
        x0, x1, x2, x3, x4, x5, x6, x7 = v
        # x^8 + 20 x^7 + 170 x^6 + 800 x^5 + 2273 x^4 + 3980 x^3 + 4180 x^2 + 2400 x + 576
        # is stable with roots in -1, -2, -3, -4
        return [x1, x2,
                x3, x4,
                x5, x6,
                x7,
                -20*x7 - 170*x6 - 800*x5 - 2273*x4 - 3980*x3 - 4180*x2 - 2400*x1 - 576*x0
                ]

    def XD(_, v):
        x0, x1, x2, x3, x4, x5, x6, x7 = v
        return _And( x0 ** 2 + x1 ** 2 + x2 ** 2 + x3 ** 2 + x4 ** 2 + x5 ** 2 + x6 ** 2 + x7 ** 2 <= 4)

    def XI(_, v):
        x0, x1, x2, x3, x4, x5, x6, x7 = v
        return (x0 - 1.0) ** 2 + (x1 - 1.0) ** 2 + \
               (x2 - 1.0) ** 2 + (x3 - 1.0) ** 2 + \
               (x4 - 1.0) ** 2 + (x5 - 1.0) ** 2 + \
               (x6 - 1.0) ** 2 + (x7 - 1.0) ** 2 <= 0.25

    def XU(_, v):
        x0, x1, x2, x3, x4, x5, x6, x7 = v
        return (x0 + 2) ** 2 + (x1 + 2) ** 2 + \
               (x2 + 2) ** 2 + (x3 + 2) ** 2 + \
               (x4 + 2) ** 2 + (x5 + 2) ** 2 + \
               (x6 + 2) ** 2 + (x7 + 2) ** 2 <= 0.16

    def SD():
        return n_dim_sphere_init_data([0, 0, 0, 0, 0, 0, 0, 0], 4, batch_size)

    epsilon = 0

    def SI():
        return n_dim_sphere_init_data([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.25, batch_size)

    def SU():
        return n_dim_sphere_init_data([-2., -2., -2., -2., -2., -2., -2., -2.], 0.16, batch_size)

    return f, XD, XI, XU, SD(), SI(), SU(), inf_bounds_n(8)


if __name__ == '__main__':
    f, XD, XI, XU, SD, SI, SU, bonds = hi_ord_8(500, {'And': z3.And, 'Or': None})
    plt.scatter(SI[:, 0], SI[:, 1], color='g', marker='x')
    plt.scatter(SU[:, 0], SU[:, 1], color='r', marker='x')
    plt.scatter(SD[:, 0], SD[:, 1], color='b')
    plt.show()

