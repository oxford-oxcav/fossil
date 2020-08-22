from z3 import *

from src.lyap.verifier.verifier import Verifier


class Z3Verifier(Verifier):
    @staticmethod
    def new_vars(n):
        return [Real('x%d' % i) for i in range(n)]

    def new_solver(self):
        return z3.Solver()

    @staticmethod
    def solver_fncts() -> {}:
        return {'And': And, 'Or': Or, 'If': If}

    def is_sat(self, res) -> bool:
        return res == sat

    def is_unsat(self, res) -> bool:
        return res == unsat

    def _solver_solve(self, solver, fml):
        solver.add(fml)
        return solver.check()

    def _solver_model(self, solver, res):
        return solver.model()

    def _model_result(self, solver, model, x, i):
        try:
            return float(model[x].as_fraction())
        except AttributeError:
            return float(model[x].approx(10).as_fraction())
        except Z3Exception:
            try:
                return float(model[x[0, 0]].as_fraction())
            except:  # when z3 finds non-rational numbers, prints them w/ '?' at the end --> approx 10 decimals
                return float(model[x[0, 0]].approx(10).as_fraction())

    def __init__(self, n_vars, equilibrium, inner_radius, outer_radius, z3_vars):
        super().__init__(n_vars, equilibrium, inner_radius, outer_radius, z3_vars)
