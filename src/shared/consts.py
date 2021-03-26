# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
from enum import Enum, auto


class LearnerType(Enum):
    NN = auto()
    Z3 = auto()
    SCIPY = auto()


class VerifierType(Enum):
    Z3 = auto()
    DREAL = auto()


class ConsolidatorType(Enum):
    NONE = auto()
    DEFAULT = auto()


class TranslatorType(Enum):
    NONE = auto()
    DEFAULT = auto()


class LearningFactors(Enum):
    QUADRATIC = auto()


class PrimerMode(Enum):
    BARRIER = auto()
    LYAPUNOV = auto()
