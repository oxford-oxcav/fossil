# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Literal

import torch


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
    RATIONAL = auto()
    # dReal only from here
    TANH = auto()
    SIGMOID = auto()
    SOFTPLUS = auto()
    COSH = auto()


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
    ROA = auto()
    RWA = auto()
    RSWA = auto()
    RWS = auto()
    RSWS = auto()
    STABLESAFE = auto()


@dataclass
class CegisConfig:
    SYSTEM: Any = None
    CERTIFICATE: CertificateType = CertificateType.LYAPUNOV
    DOMAINS: dict[str, Any] = None
    DATA: dict[str : torch.Tensor] = None
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
    FACTORS: Literal = LearningFactors.NONE
    LLO: bool = False  # last layer of ones
    ROUNDING: int = 3
    N_VARS: int = 0
    N_HIDDEN_NEURONS: tuple[int] = (10,)
    ACTIVATION: tuple[ActivationType, ...] = (ActivationType.SQUARE,)
    VERBOSE: bool = True
    ENET: Any = None
    CTRLAYER: tuple[int] = None  # not None means control certificate
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
    components_times = "components_times"
    ENet = "ENet"
    xdot = "xdot"
    xdot_func = "xdot_func"


class CegisComponentsState:
    name = "name"
    instance = "instance"
    to_next_component = "to_next_component"
