# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
import math
from aenum import Enum, NoAlias
import numpy as np

from src.shared.consts import VerifierType, LearnerType, ConsolidatorType, TranslatorType, LearningFactors, TimeDomain
from src.shared.activations import ActivationType


# prefer this over CegisConfig = Enum('CegisConfig', "...")
# to aid with the ide
class CegisConfig(Enum, settings=NoAlias):
    SP_SIMPLIFY             = False
    SP_HANDLE               = False
    SYMMETRIC_BELT          = False
    CEGIS_MAX_ITERS         = 10
    CEGIS_MAX_TIME_S        = math.inf  # in sec
    TIME_DOMAIN             = TimeDomain.CONTINUOUS 
    LEARNER                 = LearnerType.CONTINUOUS
    VERIFIER                = VerifierType.Z3
    CONSOLIDATOR            = ConsolidatorType.DEFAULT
    TRANSLATOR              = TranslatorType.CONTINUOUS
    CERTIFICATE             = None
    BATCH_SIZE              = 500
    LEARNING_RATE           = .1
    FACTORS                 = LearningFactors.NONE
    EQUILIBRIUM             = lambda n_vars: np.zeros((1, n_vars)),  # default in zero
    LLO                     = False  # last layer of ones
    ROUNDING                = 3
    N_VARS                  = 0
    N_HIDDEN_NEURONS        = [10]
    SYSTEM                  = []
    ACTIVATION              = [ActivationType.SQUARE]
    INNER_RADIUS            = 0
    OUTER_RADIUS            = 10
    INTERACTIVE_DOMAIN      = False
    POSITIVE_DOMAIN         = False
    SEED_AND_SPEED          = False
    CEGIS_PARAMETERS        = {}
    XD                      = 0
    XI                      = 0
    XU                      = 0
    VERBOSE                 = True
    ENET                    = None
    CTRLAYER                = None

    @property
    def k(self):
        return self.name

    @property
    def v(self):
        return self.value

class CegisStateKeys:
    x_v = 'x_v'
    x_v_dot = 'x_v_dot'
    x_v_map = 'x_v_map'
    x_sympy = 'x_sympy'
    x_dot_sympy = 'x_dot_sympy'
    sp_simplify = 'sp_simplify'
    sp_handle = 'sp_handle'
    S = 'S'
    S_dot = 'S_dot'
    B = 'B'
    B_dot = 'B_dot'
    optimizer = 'optimizer'
    V = 'V'
    V_dot = 'V_dot'
    cex = 'cex'  # counterexamples
    net = 'net'
    trajectory = 'trajectory'
    factors = 'factors'
    found = 'found'
    verification_timed_out = 'verification_timed_out'
    verifier_fun = 'verifier_fun'
    equilibrium = 'equilibrium'
    components_times = 'components_times'
    ENet = "ENet"
    xdot = 'xdot'
    xdot_func = 'xdot_func'


class CegisComponentsState:
    name = 'name'
    instance = 'instance'
    to_next_component = 'to_next_component'


if __name__ == '__main__':
    print(str(CegisComponentsState.name))
    print(type(str(CegisComponentsState.name)))
