from enum import Enum, auto


class LearnerType(Enum):
    NN = auto()
    Z3 = auto()
    SCIPY = auto()


class VerifierType(Enum):
    Z3 = auto()
    DREAL = auto()


class TrajectoriserType(Enum):
    NONE = auto()
    DEFAULT = auto()


class LearningFactors(Enum):
    LINEAR = auto()
    QUADRATIC = auto()
