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
