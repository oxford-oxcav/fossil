import math
from aenum import Enum, NoAlias
import numpy as np

from src.shared.consts import VerifierType, LearnerType, TrajectoriserType


# prefer this over CegisConfig = Enum('CegisConfig', "...")
# to aid with the ide
class CegisConfig(Enum, settings=NoAlias):
    SP_SIMPLIFY             = True
    SP_HANDLE               = True
    SYMMETRIC_BELT          = False
    CEGIS_MAX_ITERS         = 10
    CEGIS_MAX_TIME_S        = math.inf  # in sec
    LEARNER                 = LearnerType.NN
    VERIFIER                = VerifierType.Z3
    TRAJECTORISER           = TrajectoriserType.DEFAULT
    BATCH_SIZE              = 100
    LEARNING_RATE           = .1
    FACTORS                 = None
    EQUILIBRIUM             = lambda n_vars: np.zeros((1, n_vars)),  # default in zero
    LLO                     = False  # last layer of ones
    ROUNDING                = 3
    N_VARS                  = 0
    N_HIDDEN_NEURONS        = 0
    SYSTEM                  = []
    ACTIVATION              = 0
    INNER_RADIUS            = 0
    OUTER_RADIUS            = 0

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


class CegisComponentsState:
    name = 'name'
    instance = 'instance'
    to_next_component = 'to_next_component'


if __name__ == '__main__':
    print(str(CegisComponentsState.name))
    print(type(str(CegisComponentsState.name)))
