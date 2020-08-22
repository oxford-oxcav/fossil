import sympy as sp

from src.shared.learner import Learner


class SympySolver(Learner):

    @staticmethod
    def solver_fncts() -> {}:
        return {'And': sp.And, 'Or': sp.Or, 'If': sp.ITE,
                'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp}
