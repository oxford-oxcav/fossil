# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import matplotlib.pyplot as plt
import dreal

from experiments.benchmarks.domain_fcns import *
import experiments.benchmarks.models as models

import src.shared.control as control
from src.shared.consts import *
from src.certificate import RWS, RSWS


def rsws_demo():
    outer = 5.0
    batch_size = 1000
    f = models.NonPoly0()

    XD = Sphere([0.0, 0.0], outer)
    XI = Sphere([3.0, -3.0], 1)
    XU = Rectangle([2, 2], [3, 3])
    XG = Sphere([0.0, 0.0], 0.01)
    domains = {
        RSWS.XD: XD.generate_domain,
        RSWS.XI: XI.generate_domain,
        RSWS.XU: XU.generate_boundary,
        RSWS.XS: XU.generate_complement,
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


def rws_linear(controller):
    batch_size = 1000
    open_loop = models.Linear1()
    XD = Sphere([0.0, 0.0], 5.0)
    XS = Rectangle([-1, -1], [1, 1])
    XI = Rectangle([-0.5, -0.5], [0.5, 0.5])
    XG = Rectangle([-0.1, -0.1], [0.1, 0.1])
    domains = {
        RWS.XD: XD.generate_domain,
        RWS.XI: XI.generate_domain,
        RWS.XS: XS.generate_domain,
        RWS.dXS: XS.generate_boundary,
        RWS.XG: XG.generate_domain,
    }

    f = models.GeneralClosedLoopModel(open_loop, controller)
    data = {
        RWS.SD: XD.generate_data(batch_size),
        RWS.SI: XI.generate_data(batch_size),
        RWS.SS: XS.generate_data(batch_size),
        RWS.SG: XG.generate_data(batch_size),
    }
    return f, domains, data, inf_bounds_n(2)
