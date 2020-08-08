import sympy as sp


class SympySolver:

    @staticmethod
    def solver_fncts() -> {}:
        return {'And': sp.And, 'Or': sp.Or, 'If': sp.ITE,
                'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp}
