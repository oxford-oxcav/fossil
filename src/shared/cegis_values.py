import math
from enum import Enum, auto
import numpy as np

from src.shared.consts import VerifierType, LearnerType, TrajectoriserType


class _SelfNamedEnum(Enum):
    def _generate_next_value_(name, _, __, ___):
        return name

    @property
    def k(self):
        return self.name

    @property
    def v(self):
        return self.value


# prefer this over CegisConfig = Enum('CegisConfig', "...")
# to aid with the ide
class CegisConfig(_SelfNamedEnum):
    SP_SIMPLIFY             = True
    SP_HANDLE               = True
    SYMMETRIC_BELT          = False
    CEGIS_MAX_ITERS         = 200
    CEGIS_MAX_TIME_S        = math.inf  # in sec
    LEARNER                 = LearnerType.NN
    VERIFIER                = VerifierType.Z3
    TRAJECTORISER           = TrajectoriserType.DEFAULT
    BATCH_SIZE              = 100
    LEARNING_RATE           = .1
    FACTORS                 = None
    EQUILIBRIUM             = lambda n_vars: np.zeros((1, n_vars)),  # default in zero


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
    verifier_fun = 'verifier_fun'
    equilibrium = 'equilibrium'


class CegisComponentsState:
    name = 'name'
    instance = 'instance'
    to_next_component = 'to_next_component'


if __name__ == '__main__':
    print(str(CegisComponentsState.name))
    print(type(str(CegisComponentsState.name)))
