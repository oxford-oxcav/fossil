# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
from enum import Enum, auto


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
    BARRIER_LYAPUNOV = auto()
    LYAPUNOV = auto()
    RWS = auto()
    RSWS = auto()
    CTRLLYAP = auto()
    CTRLBARR = auto()
    CTRLRWS = auto()
    CTRLRSWS = auto()
