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
import z3
import dreal
from cvc5 import pythonic as cvpy
import numpy as np
import sympy as sp


class ActivationType(Enum):
    IDENTITY = auto()
    RELU = auto()
    LINEAR = auto()
    SQUARE = auto()
    POLY_2 = auto()
    RELU_SQUARE = auto()
    REQU = auto()
    POLY_3 = auto()
    POLY_4 = auto()
    POLY_5 = auto()
    POLY_6 = auto()
    POLY_7 = auto()
    POLY_8 = auto()
    EVEN_POLY_4 = auto()
    EVEN_POLY_6 = auto()
    EVEN_POLY_8 = auto()
    EVEN_POLY_10 = auto()
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
    CVC5 = auto()
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


class DomainNames(Enum):
    XD = "lie"
    XU = "unsafe"
    XI = "init"
    XG = "goal"
    XG_BORDER = "goal_border"
    XF = "final"
    XS = "safe"
    XS_BORDER = "safe_border"
    XNF = "not_final"
    XR = "region"

    @classmethod
    def border_sets(cls):
        return {
            cls.XS: cls.XS_BORDER,
            cls.XG: cls.XG_BORDER,
        }


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
    RAR = auto()
    CUSTOM = auto()

    @classmethod
    def get_certificate_sets(
        cls, certificate_type
    ) -> tuple[list[DomainNames], list[DomainNames]]:
        dn = DomainNames
        if certificate_type == CertificateType.LYAPUNOV:
            domains = [dn.XD]
            data = [dn.XD]
        elif certificate_type == CertificateType.BARRIER:
            domains = [dn.XD, dn.XI, dn.XU]
            data = [dn.XD, dn.XI, dn.XU]
        elif certificate_type == CertificateType.BARRIERALT:
            domains = [dn.XD, dn.XI, dn.XU]
            data = [dn.XD, dn.XI, dn.XU]
        elif certificate_type == CertificateType.ROA:
            domains = [dn.XD, dn.XI]
            data = [dn.XD, dn.XI]
        elif certificate_type in (CertificateType.RWS, CertificateType.RWA):
            domains = [dn.XD, dn.XI, dn.XS, dn.XS_BORDER, dn.XG]
            data = [dn.XD, dn.XI, dn.XU]
        elif certificate_type in (CertificateType.RSWS, CertificateType.RSWA):
            domains = [dn.XD, dn.XI, dn.XS, dn.XS_BORDER, dn.XG, dn.XG_BORDER]
            data = [dn.XD, dn.XI, dn.XU, dn.XS, dn.XG, dn.XG_BORDER]
        elif certificate_type == CertificateType.STABLESAFE:
            domains = [dn.XD, dn.XI, dn.XU]
            data = [dn.XD, dn.XI, dn.XU]
        elif certificate_type == CertificateType.RAR:
            domains = [dn.XD, dn.XI, dn.XS, dn.XS_BORDER, dn.XG, dn.XF]
            data = [dn.XD, dn.XI, dn.XU, dn.XG, dn.XF, dn.XNF]
        return domains, data

    @classmethod
    def get_required_borders(cls, certificate_type) -> dict[DomainNames, DomainNames]:
        if certificate_type in (CertificateType.RWS, CertificateType.RWA):
            return {DomainNames.XS: DomainNames.XS_BORDER}
        elif certificate_type in (CertificateType.RSWS, CertificateType.RSWA):
            return {
                DomainNames.XS: DomainNames.XS_BORDER,
                DomainNames.XG: DomainNames.XG_BORDER,
            }
        elif certificate_type == CertificateType.RAR:
            return {
                DomainNames.XS: DomainNames.XS_BORDER,
            }
        else:
            return {}


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
    N_DATA: int = 500
    LEARNING_RATE: float = 0.1
    FACTORS: Literal = LearningFactors.NONE
    LLO: bool = False  # last layer of ones
    ROUNDING: int = 3
    N_VARS: int = 0
    N_HIDDEN_NEURONS: tuple[int] = (10,)
    ACTIVATION: tuple[ActivationType, ...] = (ActivationType.SQUARE,)
    VERBOSE: int = 0
    ENET: Any = None
    CTRLAYER: tuple[int] = None  # not None means control certificate
    CTRLACTIVATION: tuple[ActivationType, ...] = None
    N_HIDDEN_NEURONS_ALT: tuple[int] = (10,)  # For DoubleCegis
    ACTIVATION_ALT: tuple[ActivationType, ...] = (
        ActivationType.SQUARE,
    )  # For DoubleCegis
    SEED: int = None
    CUSTOM_CERTIFICATE: Any = None

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


ACTIVATION_NAMES = {
    ActivationType.IDENTITY: "identity",
    ActivationType.RELU: "$ReLU$",
    ActivationType.LINEAR: "$\\varphi_{1}$",
    ActivationType.SQUARE: "$\\varphi_{2}$",
    ActivationType.POLY_2: "$\\varphi_{2}$",
    ActivationType.RELU_SQUARE: "$ReLU\\varphi_{2}$",
    ActivationType.REQU: "$ReLU\\varphi_{2}$",
    ActivationType.POLY_3: "$\\varphi_{3}$",
    ActivationType.POLY_4: "$\\varphi_{4}$",
    ActivationType.POLY_5: "$\\varphi_{5}$",
    ActivationType.POLY_6: "$\\varphi_{6}$",
    ActivationType.POLY_7: "$\\varphi_{7}$",
    ActivationType.POLY_8: "$\\varphi_{8}$",
    ActivationType.EVEN_POLY_4: "$\\varphi_{4}$",
    ActivationType.EVEN_POLY_6: "$\\varphi_{6}$",
    ActivationType.EVEN_POLY_8: "$\\varphi_{8}$",
    ActivationType.EVEN_POLY_10: "$\\varphi_{10}$",
    ActivationType.RATIONAL: "$\\varphi_{rat}$",
    ActivationType.TANH: "$\\sigma_{\\mathrm{t}}$",
    ActivationType.SIGMOID: "$\\sigma_{\\mathrm{sig}}$",
    ActivationType.SOFTPLUS: "$\\sigma_{\\mathrm{soft}}$",
    ActivationType.COSH: "$cosh$",
}

PROPERTIES = {
    CertificateType.LYAPUNOV: "Stability",
    CertificateType.ROA: "ROA",
    CertificateType.BARRIER: "Safety",
    CertificateType.BARRIERALT: "Safety",
    CertificateType.RAR: "RAR",
    CertificateType.RWA: "RWA",
    CertificateType.RSWA: "RSWA",
    CertificateType.RWS: "RWA",
    CertificateType.RSWS: "RSWA",
    CertificateType.STABLESAFE: "SWA",
}


Z3_FNCS = {
    "And": z3.And,
    "Or": z3.Or,
    "If": z3.If,
}
DREAL_FNCS = {
    "sin": dreal.sin,
    "cos": dreal.cos,
    "exp": dreal.exp,
    "And": dreal.And,
    "Or": dreal.Or,
    "If": dreal.if_then_else,
    "Not": dreal.Not,
}
MATH_FNCS = {
    "sin": np.sin,
    "cos": np.cos,
    "exp": np.exp,
}

SP_FNCS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "exp": sp.exp,
}

CVC5_FNCS = {
    "And": cvpy.And,
    "Or": cvpy.Or,
    "If": cvpy.If,
    "sin": cvpy.Sine,
    "cos": cvpy.Cosine,
    "exp": cvpy.Exponential,
}
