# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
from tkinter import Variable
from typing import Dict, Callable
from src.verifier.verifier_values import VerifierConfig

try:
    from dreal import *
except:
    print("No dreal")

from src.verifier.verifier import Verifier
from src.shared.utils import dreal_replacements, contains_object

class DRealVerifier(Verifier):
    @staticmethod
    def new_vars(n):
        return [Variable('x%d' % i) for i in range(n)]

    def new_solver(self):
        return None

    @staticmethod
    def check_type(x) -> bool:
        """
        :param x: any
        :returns: True if Dreal compatible, else false
        """
        return contains_object(x, Variable)

    @staticmethod
    def solver_fncts() -> Dict[str, Callable]:
        return {
            'sin': sin, 'cos': cos, 'exp': exp,
            'And': And, 'Or': Or, 'If': if_then_else,
            'Check': DRealVerifier.check_type,
            'Not': Not
        }

    def is_sat(self, res) -> bool:
        return isinstance(res, Box)

    @staticmethod
    def replace_point(expr, ver_vars, point):
        return dreal_replacements(expr, ver_vars, point)

    def is_unsat(self, res) -> bool:
        # int(str("x0")) = 0
        bounds_not_ok = not self.within_bounds(res)
        return res is None or bounds_not_ok

    def within_bounds(self, res) -> bool:
        return isinstance(res, Box) and all(self.in_bounds(int(str(x)[1:]), interval.mid()) for x, interval in res.items())

    def _solver_solve(self, solver, fml):
        res = CheckSatisfiability(fml, 0.00001)
        if self.is_sat(res) and not self.within_bounds(res):
            new_bound = self.optional_configs.get(VerifierConfig.DREAL_SECOND_CHANCE_BOUND.k, VerifierConfig.DREAL_SECOND_CHANCE_BOUND.v)
            fml = And(fml, *(And(x < new_bound, x > -new_bound) for x in self.xs))
            res = CheckSatisfiability(fml, 0.00001)
        return res

    def _solver_model(self, solver, res):
        assert self.is_sat(res)
        return res

    def _model_result(self, solver, model, x, idx):
        return float(model[idx].mid())

    def __init__(self, n_vars, constraints_method, vars_bounds, solver_vars, **kw):
        super().__init__(n_vars, constraints_method, vars_bounds, solver_vars, **kw)
