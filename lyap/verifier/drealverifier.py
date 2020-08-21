try:
    from dreal import *
except:
    print("No dreal")

from lyap.verifier import Verifier


class DRealVerifier(Verifier):
    @staticmethod
    def new_vars(n):
        return [Variable('x%d' % i) for i in range(n)]

    def new_solver(self):
        return None

    @staticmethod
    def solver_fncts() -> {}:
        return {
            'sin': sin, 'cos': cos, 'exp': exp,
            'And': And, 'Or': Or, 'If': if_then_else
        }

    def is_sat(self, res) -> bool:
        return isinstance(res, Box)

    def is_unsat(self, res) -> bool:
        # int(str("x0")) = 0
        # bounds_not_ok = isinstance(res, Box) and any(not self.in_bounds(int(str(x)[1:]), interval.mid()) for x, interval in res.items())
        return res is None # or bounds_not_ok

    def _solver_solve(self, solver, fml):
        return CheckSatisfiability(fml, 0.00001)

    def _solver_model(self, solver, res):
        assert self.is_sat(res)
        return res

    def _model_result(self, solver, model, x, idx):
        return float(model[idx].mid())

    def __init__(self, n_vars, equilibrium, inner_radius, outer_radius, dreal_vars):
        super().__init__(n_vars, equilibrium, inner_radius, outer_radius, dreal_vars)
