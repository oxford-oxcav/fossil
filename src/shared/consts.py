# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any
from aenum import Enum, auto, NoAlias
from dataclasses import dataclass
import math

import numpy as np


class LearnerType(Enum):
    CONTINUOUS = auto()
    DISCRETE = auto()


class VerifierType(Enum):
    Z3 = auto()
    DREAL = auto()
    MARABOU = auto()


class ConsolidatorType(Enum):
    NONE = auto()
    DEFAULT = auto()


class TranslatorType(Enum):
    DISCRETE = auto()
    CONTINUOUS = auto()
    DOUBLE = auto()


class LearningFactors(Enum):
    QUADRATIC = auto()
    NONE = auto()


class TimeDomain(Enum):
    CONTINUOUS = auto()
    DISCRETE = auto()


class PrimerMode(Enum):
    BARRIER = auto()
    LYAPUNOV = auto()


class CertificateType(Enum):
    BARRIER = auto()
    BARRIERALT = auto()
    LYAPUNOV = auto()
    RWA = auto()
    RSWA = auto()
    RWS = auto()
    RSWS = auto()
    STABLESAFE = auto()


class ActivationType(Enum):
    IDENTITY = auto()
    RELU = auto()
    LINEAR = auto()
    SQUARE = auto()
    LIN_SQUARE = auto()
    RELU_SQUARE = auto()
    REQU = auto()
    LIN_TO_CUBIC = auto()
    LIN_TO_QUARTIC = auto()
    LIN_TO_QUINTIC = auto()
    LIN_TO_SEXTIC = auto()
    LIN_TO_SEPTIC = auto()
    LIN_TO_OCTIC = auto()
    SQUARE_DEC = auto()
    # dReal only from here
    TANH = auto()
    SIGMOID = auto()
    SOFTPLUS = auto()
    COSH = auto()


# prefer this over CegisConfig = Enum('CegisConfig', "...")
# to aid with the ide
class CegisConfigOld(Enum, settings=NoAlias):
    SYSTEM = []
    SD = {}
    XD = {}
    GOAL = {}
    SP_SIMPLIFY = False
    SP_HANDLE = False
    SYMMETRIC_BELT = False
    CEGIS_MAX_ITERS = 10
    CEGIS_MAX_TIME_S = math.inf  # in sec
    TIME_DOMAIN = TimeDomain.CONTINUOUS
    LEARNER = LearnerType.CONTINUOUS
    VERIFIER = VerifierType.Z3
    CONSOLIDATOR = ConsolidatorType.DEFAULT
    TRANSLATOR = TranslatorType.CONTINUOUS
    CERTIFICATE = None
    BATCH_SIZE = 500
    LEARNING_RATE = 0.1
    FACTORS = LearningFactors.NONE
    EQUILIBRIUM = (lambda n_vars: np.zeros((1, n_vars)),)  # default in zero
    LLO = False  # last layer of ones
    ROUNDING = 3
    N_VARS = 0
    N_HIDDEN_NEURONS = [10]
    ACTIVATION = [ActivationType.SQUARE]
    INNER_RADIUS = 0
    OUTER_RADIUS = 10
    INTERACTIVE_DOMAIN = False
    POSITIVE_DOMAIN = False
    SEED_AND_SPEED = False
    CEGIS_PARAMETERS = {}
    VERBOSE = True
    ENET = None
    CTRLAYER = None
    CTRLACTIVATION = None
    N_HIDDEN_NEURONS_ALT = [10]  # For DoubleCegis
    ACTIVATION_ALT = [ActivationType.SQUARE]  # For DoubleCegis

    @property
    def k(self):
        return self.name

    @property
    def v(self):
        return self.value


@dataclass
class CegisConfig:
    SYSTEM: Any = None
    CERTIFICATE: CertificateType = CertificateType.LYAPUNOV
    SD: Any = None
    XD: Any = None
    GOAL: Any = None
    SP_SIMPLIFY: bool = False
    SP_HANDLE: bool = False
    SYMMETRIC_BELT: bool = False
    CEGIS_MAX_ITERS: int = 10
    CEGIS_MAX_TIME_S: float = math.inf  # in sec
    TIME_DOMAIN: TimeDomain = TimeDomain.CONTINUOUS
    LEARNER: LearnerType = LearnerType.CONTINUOUS
    VERIFIER: VerifierType = VerifierType.Z3
    CONSOLIDATOR: ConsolidatorType = ConsolidatorType.DEFAULT
    TRANSLATOR: TranslatorType = TranslatorType.CONTINUOUS
    BATCH_SIZE: int = 500
    LEARNING_RATE: float = 0.1
    FACTORS = LearningFactors.NONE
    EQUILIBRIUM: Any = (lambda n_vars: np.zeros((1, n_vars)),)  # default in zero
    LLO: bool = False  # last layer of ones
    ROUNDING: int = 3
    N_VARS: int = 0
    N_HIDDEN_NEURONS: tuple[int] = (10,)
    ACTIVATION: tuple[ActivationType, ...] = (ActivationType.SQUARE,)
    INNER_RADIUS: float = 0
    OUTER_RADIUS: float = 10
    INTERACTIVE_DOMAIN: bool = False
    POSITIVE_DOMAIN: bool = False
    SEED_AND_SPEED: bool = False
    VERBOSE: bool = True
    ENET: Any = None
    CTRLAYER: tuple[int] = None
    CTRLACTIVATION: tuple[ActivationType, ...] = None
    N_HIDDEN_NEURONS_ALT: tuple[int] = (10,)  # For DoubleCegis
    ACTIVATION_ALT: tuple[ActivationType, ...] = (
        ActivationType.SQUARE,
    )  # For DoubleCegis

    def __getitem__(self, item):
        return getattr(self, item)


class CegisStateKeys:
    x_v = "x_v"
    x_v_dot = "x_v_dot"
    x_v_map = "x_v_map"
    x_sympy = "x_sympy"
    x_dot_sympy = "x_dot_sympy"
    sp_simplify = "sp_simplify"
    sp_handle = "sp_handle"
    S = "S"
    S_dot = "S_dot"
    B = "B"
    B_dot = "B_dot"
    optimizer = "optimizer"
    V = "V"
    V_dot = "V_dot"
    cex = "cex"  # counterexamples
    net = "net"
    trajectory = "trajectory"
    factors = "factors"
    found = "found"
    verification_timed_out = "verification_timed_out"
    verifier_fun = "verifier_fun"
    equilibrium = "equilibrium"
    components_times = "components_times"
    ENet = "ENet"
    xdot = "xdot"
    xdot_func = "xdot_func"


class CegisComponentsState:
    name = "name"
    instance = "instance"
    to_next_component = "to_next_component"
