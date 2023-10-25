# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import dreal
import matplotlib.pyplot as plt

import experiments.benchmarks.models as models
import fossil.certificate as certificate
import fossil.control as control
from fossil.consts import *
from fossil.domains import *


def rsws_demo():
    outer = 5.0
    batch_size = 1000
    f = models.NonPoly0()

    XD = Sphere([0.0, 0.0], outer)
    XI = Sphere([3.0, -3.0], 1)
    XU = Rectangle([2, 2], [3, 3])
    XG = Sphere([0.0, 0.0], 0.01)
    domains = {
        certificate.XD: XD.generate_domain,
        certificate.XI: XI.generate_domain,
        certificate.XU: XU.generate_boundary,
        certificate.XS: XU.generate_complement,
        certificate.XG: XG.generate_domain,
        certificate.XG_BORDER: XG.generate_boundary,
    }
    data = {
        certificate.XD: XD.generate_data(batch_size),
        certificate.XI: XI.generate_data(batch_size),
        certificate.XU: XU.generate_data(batch_size),
        certificate.XG: XG.generate_data(batch_size),
    }
    return f, domains, data, inf_bounds_n(2)


def rws_linear(controller):
    batch_size = 1000
    open_loop = models.Linear1()
    XD = Sphere([0.0, 0.0], 5.0)
    XS = Rectangle([-1, -1], [1, 1])
    XI = Rectangle([-0.5, -0.5], [0.5, 0.5])
    XG = Rectangle([-0.1, -0.1], [0.1, 0.1])
    domains = {
        certificate.XD: XD.generate_domain,
        certificate.XI: XI.generate_domain,
        certificate.XS: XS.generate_domain,
        certificate.XS_BORDER: XS.generate_boundary,
        certificate.XG: XG.generate_domain,
    }

    f = models.GeneralClosedLoopModel(open_loop, controller)
    data = {
        certificate.XD: XD.generate_data(batch_size),
        certificate.XI: XI.generate_data(batch_size),
        certificate.XS: XS.generate_data(batch_size),
        certificate.XG: XG.generate_data(batch_size),
    }
    return f, domains, data, inf_bounds_n(2)
