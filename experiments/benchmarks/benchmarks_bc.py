# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import matplotlib.pyplot as plt
from pyrsistent import b
from z3 import z3
import dreal

from experiments.benchmarks.domain_fcns import *
import experiments.benchmarks.models as models
from src.shared.activations import ActivationType
import src.shared.control as control
from src.certificate import Barrier, BarrierLyapunov

# this series comes from
# Synthesizing Barrier Certificates Using Neural Networks
# by Zhao H. et al
# HSCC 2020

inf = 1e300
inf_bounds = [-inf, inf]


def inf_bounds_n(n):
    return [inf_bounds] * n


def barr_3():
    batch_size = 500
    f = models.Barr3()

    XD = Rectangle([-3, -2], [2.5, 1])

    class Init(Set):
        def generate_domain(self, v):
            x, y = v
            f = self.set_functions(v)
            _Or = f["Or"]
            _And = f["And"]
            return _Or(
                _And((x - 1.5) ** 2 + y ** 2 <= 0.25),
                _And(x >= -1.8, x <= -1.2, y >= -0.1, y <= 0.1),
                _And(x >= -1.4, x <= -1.2, y >= -0.5, y <= 0.1),
            )

        def generate_data(self, batch_size):
            n0 = int(batch_size / 3)
            n1 = n0
            n2 = batch_size - (n0 + n1)
            return torch.cat(
                [
                    circle_init_data((1.5, 0.0), 0.25, n0),
                    square_init_data([[-1.8, -0.1], [-1.2, 0.1]], n1),
                    add_corners_2d([[-1.8, -0.1], [-1.2, 0.1]]),
                    square_init_data([[-1.4, -0.5], [-1.2, 0.1]], n2),
                    add_corners_2d([[-1.4, -0.5], [-1.2, 0.1]]),
                ]
            )

    class Unsafe(Set):
        def generate_domain(self, v):
            x, y = v
            f = self.set_functions(v)
            _Or = f["Or"]
            _And = f["And"]
            return _Or(
                (x + 1) ** 2 + (y + 1) ** 2 <= 0.16,
                _And(0.4 <= x, x <= 0.6, 0.1 <= y, y <= 0.5),
                _And(0.4 <= x, x <= 0.8, 0.1 <= y, y <= 0.3),
            )

        def generate_data(self, batch_size):
            n0 = int(batch_size / 3)
            n1 = n0
            n2 = batch_size - (n0 + n1)
            return torch.cat(
                [
                    circle_init_data((-1.0, -1.0), 0.16, n0),
                    square_init_data([[0.4, 0.1], [0.6, 0.5]], n1),
                    add_corners_2d([[0.4, 0.1], [0.6, 0.5]]),
                    square_init_data([[0.4, 0.1], [0.8, 0.3]], n2),
                    add_corners_2d([[0.4, 0.1], [0.8, 0.3]]),
                ]
            )

    XI = Init()

    XU = Unsafe()

    domains = {
        "lie": XD.generate_domain,
        "init": XI.generate_domain,
        "unsafe": XU.generate_domain,
    }
    data = {
        "lie": XD.generate_data(batch_size),
        "init": XI.generate_data(batch_size),
        "unsafe": XU.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(2)


def barr_1():
    _And = dreal.And
    batch_size = 500

    f = models.Barr1()

    class UnsafeDomain(Set):
        def generate_domain(self, v):
            x, y = v
            return x + y ** 2 <= 0

        def generate_data(self, batch_size):
            points = []
            limits = [[-2, -2], [0, 2]]
            while len(points) < batch_size:
                dom = square_init_data(limits, batch_size)
                idx = torch.nonzero(dom[:, 0] + dom[:, 1] ** 2 <= 0)
                points += dom[idx][:, 0, :]
            return torch.stack(points[:batch_size])

    XD = Rectangle([-2, -2], [2, 2])
    XI = Rectangle([0, 1], [1, 2])
    XU = UnsafeDomain()
    domains = {
        "lie": XD.generate_domain,
        "init": XI.generate_domain,
        "unsafe": XU.generate_domain,
    }

    data = {
        "lie": XD.generate_data(batch_size),
        "init": XI.generate_data(batch_size),
        "unsafe": XU.generate_data(batch_size),
    }
    return f, domains, data, inf_bounds_n(2)


def barr_2():
    batch_size = 500

    f = models.Barr2()

    # This might be a terrible way to do this
    class Domain(Set):
        def generate_domain(self, v):
            x, y = v
            f = self.set_functions(v)
            return f["And"](-2 <= x, y <= 2)

        def generate_data(self, batch_size):
            x_comp = -2 + torch.randn(batch_size, 1) ** 2
            y_comp = 2 - torch.randn(batch_size, 1) ** 2
            dom = torch.cat([x_comp, y_comp], dim=1)
            return dom

    XD = Domain()
    XI = Sphere([-0.5, 0.5], 0.4)
    XU = Sphere([0.7, -0.7], 0.3)

    domains = {
        "lie": XD.generate_domain,
        "init": XI.generate_domain,
        "unsafe": XU.generate_domain,
    }
    data = {
        "lie": XD.generate_data(batch_size),
        "init": XI.generate_data(batch_size),
        "unsafe": XU.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(2)


def obstacle_avoidance():
    batch_size = 1000
    f = models.ObstacleAvoidance()

    class Domain(Set):
        def generate_domain(self, v):
            x, y, phi = v
            f = self.set_functions(v)
            return f["And"](-2 <= x, x <= 2, -2 <= y, y <= 2, -1.57 <= phi, phi <= 1.57)

        def generate_data(self, batch_size):
            k = 4
            x_comp = -2 + torch.sum(torch.randn(batch_size, k) ** 2, dim=1).reshape(
                batch_size, 1
            )
            y_comp = 2 - torch.sum(torch.randn(batch_size, k) ** 2, dim=1).reshape(
                batch_size, 1
            )
            phi_comp = segment([-1.57, 1.57], batch_size)
            dom = torch.cat([x_comp, y_comp, phi_comp], dim=1)
            return dom

    class Init(Set):
        def generate_domain(self, v):
            x, y, phi = v
            f = self.set_functions(v)
            return f["And"](
                -0.1 <= x, x <= 0.1, -2 <= y, y <= -1.8, -0.52 <= phi, phi <= 0.52
            )

        def generate_data(self, batch_size):
            x = segment([-0.1, 0.1], batch_size)
            y = segment([-2.0, -1.8], batch_size)
            phi = segment([-0.52, 0.52], batch_size)
            return torch.cat([x, y, phi], dim=1)

    class UnsafeDomain(Set):
        def generate_domain(self, v):
            x, y, _phi = v
            return x ** 2 + y ** 2 <= 0.04

        def generate_data(self, batch_size):
            xy = circle_init_data((0.0, 0.0), 0.04, batch_size)
            phi = segment([-0.52, 0.52], batch_size)
            return torch.cat([xy, phi], dim=1)

    XD = Domain()
    XI = Init()
    XU = UnsafeDomain()
    domains = {
        "lie": XD.generate_domain,
        "init": XI.generate_domain,
        "unsafe": XU.generate_domain,
    }
    data = {
        "lie": XD.generate_data(batch_size),
        "init": XI.generate_data(batch_size),
        "unsafe": XU.generate_data(batch_size),
    }
    bounds = inf_bounds_n(2)
    pi = math.pi
    bounds.append([-pi / 2, pi / 2])
    return f, domains, data, bounds


def twod_hybrid():
    batch_size = 500
    # A = [ [0,1,0], [0,0,1] [-0.2,-0.3,-1] ]
    # B = [0, 0, 0.1]
    # C = [1, 0, 0]  --> output is x0
    # input : if y >= 0: u=10, else: u=-10
    # X: x0**2 + x1**2 + x2**2 <= 16
    # Xi (x0+2)**2 + x1**2 + x2**2 <= 0.01
    # Xi (x0-2)**2 + x1**2 + x2**2 <= 0.01

    f = models.TwoD_Hybrid()

    XD = Sphere([0, 0], 2)
    XI = Sphere([-1, -1], 0.5)
    XU = Sphere([1, 1], 0.5)
    domains = {
        "lie": XD.generate_domain,
        "init": XI.generate_domain,
        "unsafe": XU.generate_domain,
    }
    data = {
        "lie": XD.generate_data(batch_size),
        "init": XI.generate_data(batch_size),
        "unsafe": XU.generate_data(batch_size),
    }
    bounds = inf_bounds_n(2)
    return f, domains, data, bounds


# 4-d ODE benchmark
def hi_ord_4():
    batch_size = 1000
    f = models.HighOrd4()
    XD = Sphere([0, 0, 0, 0], 2)
    XI = Sphere([1, 1, 1, 1], 0.5)
    XU = Sphere([-2, -2, -2, -2], 0.4)

    domains = {
        "lie": XD.generate_domain,
        "init": XI.generate_domain,
        "unsafe": XU.generate_domain,
    }
    data = {
        "lie": XD.generate_data(batch_size),
        "init": XI.generate_data(batch_size),
        "unsafe": XU.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(4)


# 6-d ODE benchmark
def hi_ord_6():
    batch_size = 1000
    f = models.HighOrd6()

    XD = Sphere([0, 0, 0, 0, 0, 0], 2)
    XI = Sphere([1, 1, 1, 1, 1, 1], 0.5)
    XU = Sphere([-2, -2, -2, -2, -2, -2], 0.4)

    domains = {
        "lie": XD.generate_domain,
        "init": XI.generate_domain,
        "unsafe": XU.generate_domain,
    }
    data = {
        "lie": XD.generate_data(batch_size),
        "init": XI.generate_data(batch_size),
        "unsafe": XU.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(6)


# 8-d ODE benchmark
def hi_ord_8():
    batch_size = 1000
    f = models.HighOrd8()

    XD = Sphere([0, 0, 0, 0, 0, 0, 0, 0], 2)
    XI = Sphere([1, 1, 1, 1, 1, 1, 1, 1], 0.5)
    XU = Sphere([-2, -2, -2, -2, -2, -2, -2, -2], 0.4)

    domains = {
        "lie": XD.generate_domain,
        "init": XI.generate_domain,
        "unsafe": XU.generate_domain,
    }
    data = {
        "lie": XD.generate_data(batch_size),
        "init": XI.generate_data(batch_size),
        "unsafe": XU.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(8)


def safe_control_ct():
    outer = 1
    batch_size = 1000

    open_loop = models.UnstableLinear()
    XD = Torus([0.0, 0.0], outer, 0.1)
    XI = Sphere([0.7, 0.7], 0.2)
    XU = Sphere([-0.7, -0.7], 0.2)
    ctrler = control.SafeStableCT(2, [1], [ActivationType.LINEAR], XU)
    optim = torch.optim.AdamW(ctrler.parameters())
    ctrler.learn(XD.generate_data(batch_size), open_loop, optim)
    f = models.ClosedLoopModel(open_loop, ctrler)

    domains = {
        "lie": XD.generate_domain,
        "init": XI.generate_domain,
        "unsafe": XU.generate_domain,
    }
    data = {
        "lie": XD.generate_data(batch_size),
        "init": XI.generate_data(batch_size),
        "unsafe": XU.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(2)


def car_control():
    outer = 1
    batch_size = 1000
    open_loop = models.Car()
    XD = Torus([0.0, 0.0, 0.0], outer, 0.1)
    XI = Sphere([0.7, 0.7, 0.7 ], 0.2)
    XU = Sphere([-0.7, -0.7, -0.7], 0.2)
    ctrler = control.SafeStableCT(3, [1], [ActivationType.LINEAR], XU)
    optim = torch.optim.AdamW(ctrler.parameters())
    ctrler.learn(XD.generate_data(batch_size), open_loop, optim)
    f = models.ClosedLoopModel(open_loop, ctrler)

    domains = {
        "lie": XD.generate_domain,
        "init": XI.generate_domain,
        "unsafe": XU.generate_domain,
    }
    data = {
        "lie": XD.generate_data(batch_size),
        "init": XI.generate_data(batch_size),
        "unsafe": XU.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(3)

if __name__ == "__main__":
    f, X, S,  bounds = safe_control_ct()
    from src.plots.plot_fcns import vector_field
    from matplotlib import pyplot as plt
    torch.manual_seed(169)
    xx = np.linspace(-10, 10, 20)
    yy = np.linspace(-10, 10, 20)
    XX, YY = np.meshgrid(xx, yy)
    ax = vector_field(f, XX, YY, None)
    plt.show()
