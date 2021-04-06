# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
from z3 import *

from src.lyap.verifier.verifier import Verifier
from src.lyap.utils import z3_replacements


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

    @staticmethod
    def replace_point(expr, ver_vars, point):
        return z3_replacements(expr, ver_vars, point)

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

    def __init__(self, n_vars, equilibrium, domain, z3_vars, **kw):
        super().__init__(n_vars, equilibrium, domain, z3_vars, **kw)
