# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch

import sympy as sp
import re
from matplotlib import pyplot as plt

from experiments.benchmarks.domain_fcns import *
import experiments.benchmarks.models as models
from src.shared.activations import ActivationType
import src.shared.control as control
from src.certificate import Lyapunov, RSWS, Barrier

###############################
# NON POLY BENCHMARKS
###############################

# this series comes from
# 2014, Finding Non-Polynomial Positive Invariants and Lyapunov Functions for
# Polynomial Systems through Darboux Polynomials.

# also from CDC 2011, Parrillo, poly system w non-poly lyap


def nonpoly0_lyap():
    p = models.NonPoly0()
    domain = Torus([0, 0], 10, 0.1)

    return (
        p,
        {Lyapunov.XD: domain.generate_domain},
        {Lyapunov.SD: domain.generate_data(1000)},
        inf_bounds_n(2),
    )


def nonpoly0_rws():
    p = models.NonPoly0()
    XD = Sphere([0, 0], 10)
    goal = Sphere([0, 0], 0.1)
    unsafe = Sphere([3, 3], 0.5)
    init = Sphere([-3, -3], 0.5)
    batch_size = 500
    domains = {
        "lie": XD.generate_domain,
        "init": init.generate_domain,
        "unsafe": unsafe.generate_boundary,
        "goal": goal.generate_domain,
    }

    data = {
        "lie": SetMinus(XD, goal).generate_data(batch_size),
        "init": init.generate_data(batch_size),
        "unsafe": unsafe.generate_data(batch_size),
    }

    return p, domains, data, inf_bounds_n(2)


def nonpoly1():

    outer = 10.0
    batch_size = 500

    f = models.NonPoly1()

    XD = PositiveOrthantSphere([0.0, 0.0], outer)

    domains = {
        Lyapunov.XD: XD.generate_domain,
    }

    data = {
        Lyapunov.SD: XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(2)


def nonpoly2():

    outer = 10.0
    batch_size = 750

    f = models.NonPoly2()

    XD = PositiveOrthantSphere([0.0, 0.0, 0.0], outer)

    domains = {
        Lyapunov.XD: XD.generate_domain,
    }

    data = {
        Lyapunov.SD: XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(3)


def nonpoly3():

    outer = 10.0
    batch_size = 500

    f = models.NonPoly3()

    XD = PositiveOrthantSphere([0.0, 0.0, 0.0], outer)

    domains = {
        Lyapunov.XD: XD.generate_domain,
    }

    data = {
        Lyapunov.SD: XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(3)


# POLY benchmarks


def benchmark_0():

    outer = 10.0
    batch_size = 1000
    # test function, not to be included
    f = models.Benchmark0()

    XD = Sphere([0.0, 0.0], outer)

    domains = {
        Lyapunov.XD: XD.generate_domain,
    }

    data = {
        Lyapunov.SD: XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(2)


def poly_1():

    outer = 10.0
    inner = 0.1
    batch_size = 500
    # SOSDEMO2
    # from http://sysos.eng.ox.ac.uk/sostools/sostools.pdf
    f = models.Poly1()

    XD = Torus([0.0, 0.0, 0.0], outer, inner)

    domains = {
        Lyapunov.XD: XD.generate_domain,
    }

    data = {
        Lyapunov.SD: XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(3)


# this series comes from
# https://www.cs.colorado.edu/~srirams/papers/nolcos13.pdf
# srirams paper from 2013 (old-ish) but plenty of lyap fcns
def poly_2():

    outer = 10.0
    inner = 0.01
    batch_size = 500

    f = models.Poly2()

    XD = Torus([0.0, 0.0], outer, inner)

    domains = {
        Lyapunov.XD: XD.generate_domain,
    }

    data = {
        Lyapunov.SD: XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(2)


def poly_3():

    outer = 10.0
    inner = 0.1
    batch_size = 500

    f = models.Poly3()

    XD = Torus([0.0, 0.0], outer, inner)

    domains = {
        Lyapunov.XD: XD.generate_domain,
    }

    data = {
        Lyapunov.SD: XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(2)


def poly_4():

    outer = 10.0
    inner = 0.1
    batch_size = 500

    f = models.Poly4()

    XD = Torus([0.0, 0.0], outer, inner)

    domains = {
        Lyapunov.XD: XD.generate_domain,
    }

    data = {
        Lyapunov.SD: XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(2)


def twod_hybrid():

    outer = 10.0
    inner = 0.01
    batch_size = 1000
    # example of 2-d hybrid sys
    f = models.TwoDHybrid()

    XD = Torus([0.0, 0.0], outer, inner)

    domains = {
        Lyapunov.XD: XD.generate_domain,
    }

    data = {
        Lyapunov.SD: XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(2)


def linear_discrete():

    outer = 10.0
    inner = 0.01
    batch_size = 500

    f = models.LinearDiscrete()

    XD = Torus([0.0, 0.0], outer, inner)

    domains = {
        Lyapunov.XD: XD.generate_domain,
    }

    data = {
        Lyapunov.SD: XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(2)


def double_linear_discrete():

    outer = 10.0
    batch_size = 1000
    f = models.DoubleLinearDiscrete()

    XD = Sphere([0.0, 0.0, 0.0, 0.0], outer)

    domains = {
        Lyapunov.XD: XD.generate_domain,
    }

    data = {
        Lyapunov.SD: XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(4)


def linear_discrete_n_vars(smt_verification, n_vars):

    outer = 10.0
    batch_size = 1000

    f = models.LinearDiscreteNVars()

    XD = Sphere([0.0] * n_vars, outer)

    data = {Lyapunov.SD: XD.generate_data(batch_size)}

    if smt_verification:
        domains = {Lyapunov.XD: XD.generate_domain}
    else:
        lower_inputs = -outer * np.ones((1, n_vars))
        upper_inputs = outer * np.ones((1, n_vars))
        initial_bound = jax_verify.IntervalBound(lower_inputs, upper_inputs)
        domains = {Lyapunov.XD: initial_bound}

    return f, domains, data, inf_bounds_n(n_vars)


def non_linear_discrete():

    outer = 10.0
    batch_size = 1000

    f = models.NonLinearDiscrete()

    XD = Sphere([0.0, 0.0], outer)

    domains = {
        Lyapunov.XD: XD.generate_domain,
    }

    data = {
        Lyapunov.SD: XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(2)


def rsws_demo():
    outer = 5.0
    batch_size = 1000
    f = models.NonPoly0()

    XD = Sphere([0.0, 0.0], outer)
    XI = Sphere([3.0, -3.0], 1)
    XU = Rectangle([-1, -5], [5, 1])
    XG = Sphere([0.0, 0.0], 0.01)
    domains = {
        RSWS.XD: XD.generate_domain,
        RSWS.XI: XI.generate_domain,
        RSWS.XU: XU.generate_boundary,
        RSWS.XS: XU.generate_completement,
        RSWS.XG: XG.generate_domain,
        RSWS.dXG: XG.generate_boundary,
    }
    data = {
        RSWS.SD: XD.generate_data(batch_size),
        RSWS.SI: XI.generate_data(batch_size),
        RSWS.SU: XU.generate_data(batch_size),
        RSWS.SG: XG.generate_data(batch_size),
    }
    return f, domains, data, inf_bounds_n(2)


def ras_demo_lyap():
    outer = 5.0
    inner = 0.01
    batch_size = 1000
    f = models.NonPoly0()

    XD = Torus([0.0, 0.0], outer, inner)
    XI = Sphere([3.0, -3.0], 1)
    XU = Rectangle([-1, -5], [5, 1])
    XG = Sphere([0.0, 0.0], 0.01)
    domains_lyap = {
        Lyapunov.XD: XD.generate_domain,
    }
    data_lyap = {
        Lyapunov.XD: XD.generate_data(batch_size),
    }

    return f, domains_lyap, data_lyap, inf_bounds_n(2)


def ras_demo_barr():
    outer = 5.0
    inner = 0.01
    batch_size = 1000
    f = models.NonPoly0()
    XD = Torus([0.0, 0.0], outer, inner)
    XI = Sphere([3.0, -3.0], 1)
    XU = Rectangle([-1, -5], [5, 1])
    XG = Sphere([0.0, 0.0], 0.01)
    domains_barr = {
        Barrier.XD: XD.generate_domain,
        Barrier.XI: XI.generate_domain,
        Barrier.XU: XU.generate_domain,
    }
    data_barr = {
        Barrier.XD: XD.generate_data(batch_size),
        Barrier.XI: XI.generate_data(batch_size),
        Barrier.XU: XU.generate_data(batch_size),
    }

    return f, domains_barr, data_barr, inf_bounds_n(2)


def control_ct():
    outer = 10
    batch_size = 1000

    open_loop = models.UnstableLinear()
    XD = Torus([0.0, 0.0], outer, 0.1)
    ctrler = control.StabilityCT(dim=2, layers=[1], activations=[ActivationType.LINEAR])
    optim = torch.optim.AdamW(ctrler.parameters())
    ctrler.learn(XD.generate_data(batch_size), open_loop, optim)
    f = models.ClosedLoopModel(open_loop, ctrler)

    domains = {
        Lyapunov.XD: XD.generate_domain,
    }

    data = {
        Lyapunov.SD: XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(2)


def control_dt():
    outer = 10
    batch_size = 1000

    open_loop = models.UnstableLinear()
    XD = Torus([0.0, 0.0], outer, 0.1)
    ctrler = control.StabilityDT(dim=2, layers=[1], activations=[ActivationType.LINEAR])
    optim = torch.optim.AdamW(ctrler.parameters())
    ctrler.learn(XD.generate_data(batch_size), open_loop, optim)
    f = models.ClosedLoopModel(open_loop, ctrler)

    domains = {
        Lyapunov.XD: XD.generate_domain,
    }

    data = {
        Lyapunov.SD: XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(2)


def max_degree_fx(fx):
    return max(max_degree_poly(f) for f in fx)


def max_degree_poly(p):
    s = str(p)
    s = re.sub(r"x\d+", "x", s)
    try:
        f = sp.sympify(s)
        return sp.degree(f)
    except:
        print("Exception in %s for %s" % (max_degree_poly.__name__, p))
        return 0
