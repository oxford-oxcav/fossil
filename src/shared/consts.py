from enum import Enum


class LearnerType(Enum):
    NN = 0
    Z3 = 1
    SCIPY = 2


class VerifierType(Enum):
    Z3 = 0
    DREAL = 1
