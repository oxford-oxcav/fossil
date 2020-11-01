from src.shared.verifier_values import VerifierConfig

try:
    from dreal import *
except:
    print("No dreal")

from src.lyap.verifier.verifier import Verifier
from src.lyap.utils import dreal_replacements

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

    @staticmethod
    def replace_point(expr, ver_vars, point):
        return dreal_replacements(expr, ver_vars, point)

    def _solver_solve(self, solver, fml):
        res = CheckSatisfiability(fml, 0.00001)
        if not self.is_sat(res):
            new_bound = self.optional_configs.get(VerifierConfig.DREAL_SECOND_CHANCE_BOUND.k, VerifierConfig.DREAL_SECOND_CHANCE_BOUND.v)
            fml = And(fml, *(x < new_bound for x in self.xs))
            res = CheckSatisfiability(fml, 0.00001)
        return res
    def _solver_model(self, solver, res):
        assert self.is_sat(res)
        return res

    def _model_result(self, solver, model, x, idx):
        return float(model[idx].mid())

    def __init__(self, n_vars, equilibrium, domain, dreal_vars, **kw):
        super().__init__(n_vars, equilibrium, domain, dreal_vars, **kw)
