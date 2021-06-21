# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
import torch
import numpy as np
import timeit

from src.lyap.utils import Timer, timer, z3_replacements
from src.shared.cegis_values import CegisConfig, CegisStateKeys
from src.shared.component import Component
from src.shared.utils import vprint

T = Timer()


class Verifier(Component):
    def __init__(self, n_vars, equilibrium, domain, solver_vars, **kw):
        super().__init__()
        self.iter = -1
        self.n = n_vars
        self.eq = equilibrium
        self.counterexample_n = 20
        self.domain = domain
        self._last_cex = []
        self._n_cex_to_keep = self.counterexample_n * 1
        self.xs = solver_vars
        self._solver_timeout = 60
        self._vars_bounds = 1e300
        self.verbose = kw.get(CegisConfig.VERBOSE.k, CegisConfig.VERBOSE.v)
        self.optional_configs = kw

        assert self.counterexample_n > 0

    @staticmethod
    def new_vars(n):
        """Example: return [Real('x%d' % i) for i in range(n_vars)]"""
        raise NotImplementedError('')

    @staticmethod
    def solver_fncts() -> {}:
        """Example: return {'And': z3.And}"""
        raise NotImplementedError('')

    @staticmethod
    def replace_point():
        """Example: return V(ctx)"""
        raise NotImplementedError('')

    def new_solver(self):
        """Example: return z3.Solver()"""
        raise NotImplementedError('')

    def is_sat(self, res) -> bool:
        """Example: return res == sat"""
        raise NotImplementedError('')

    def is_unsat(self, res) -> bool:
        """Example: return res == unsat"""
        raise NotImplementedError('')

    def _solver_solve(self, solver, fml):
        """Example: solver.add(fml); return solver.check()"""
        raise NotImplementedError('')

    def _solver_model(self, solver, res):
        """Example: return solver.model()"""
        raise NotImplementedError('')

    def _model_result(self, solver, model, var, idx):
        """Example: return float(model[var[0, 0]].as_fraction())"""
        raise NotImplementedError('')

    def get(self, **kw):
        return self.verify(kw[CegisStateKeys.V], kw[CegisStateKeys.V_dot])

    @property
    def timeout(self):
        return self._solver_timeout

    @timeout.setter
    def timeout(self, _t):
        t = int(_t)
        assert t >= 0
        self._solver_timeout = t

    @timer(T)
    def verify(self, V, dV):
        """
        :param V: z3 expr
        :param dV: z3 expr
        :return:
                found_lyap: True if V is valid
                C: a list of ctx
        """
        found, timed_out = False, False
        C = []

        s = self.new_solver()
        fmls = self.domain_constraints(V, dV)

        # if sat, found counterexample; if unsat, V is lyap
        res, timedout = self.solve_with_timeout(s, fmls)
        if timedout:
            print(":/ timed out")
            timed_out = True
        elif self.is_unsat(res):
            vprint(['No counterexamples found!'], self.verbose)
            found = True
        else:
            original_point = self.compute_model(s, res)
            value_in_ctx, value_in_dV = self.replace_point(V, self.xs, original_point.numpy().T), \
                                          self.replace_point(dV, self.xs, original_point.numpy().T)
            vprint(['V(ctx) = ', value_in_ctx], self.verbose)
            vprint(['dV(ctx) = ', value_in_dV], self.verbose)
            C = self.randomise_counterex(original_point)

        return {CegisStateKeys.found: found, CegisStateKeys.verification_timed_out: timed_out,
                CegisStateKeys.cex: C}

    def domain_constraints(self, V, dV):
        """
        :param V:
        :param dV:
        :return:
        """
        _Or = self.solver_fncts()['Or']
        _And = self.solver_fncts()['And']

        lyap_negated = _Or(V <= 0, dV > 0)

        # domain_constr = []
        # for idx in range(self.eq.shape[0]):
        #     circle = self.circle_constr(self.eq[idx, :])
        #     domain_constr += [_And(circle > self.inner ** 2, circle < self.outer ** 2)]
        # # _And(*(domain_constr for _ in range(len(domain_constr))))
        return _And(self.domain, lyap_negated)

    def circle_constr(self, c):
        """
        :param x:
        :param c:
        :return:
        """
        circle_constr = np.sum([(x - c[i]) ** 2 for i, x in enumerate(self.xs)])

        return circle_constr

    def solve_with_timeout(self, solver, fml):
        """
        :param fml:
        :param solver: z3 solver
        :return:
                res: sat if found ctx
                timedout: true if verification timed out
        """
        try:
            solver.set("timeout", max(1, self._solver_timeout * 1000))
        except:
            pass
        timer = timeit.default_timer()
        res = self._solver_solve(solver, fml)
        timer = timeit.default_timer() - timer
        timedout = timer >= self._solver_timeout
        return res, timedout

    def compute_model(self, solver, res):
        """
        :param solver: z3 solver
        :return: tensor containing single ctx
        """
        model = self._solver_model(solver, res)
        vprint(['Counterexample Found: {}'.format(model)], self.verbose)
        temp = []
        for i, x in enumerate(self.xs):
            temp += [self._model_result(solver, model, x, i)]

        original_point = torch.tensor(temp)
        return original_point[None, :]

    # given one ctx, useful to sample around it to increase data set
    # these points might *not* be real ctx, but probably close to invalidity condition
    def randomise_counterex(self, point):
        """
        :param point: tensor
        :return: list of ctx
        """
        C = []
        # dimensionality issue
        shape = (1, max(point.shape[0], point.shape[1]))
        point = point.reshape(shape)
        for i in range(self.counterexample_n):
            random_point = point + 5*1e-3 * torch.randn(shape)
            # if self.inner < torch.norm(random_point) < self.outer:
            C.append(random_point)
        C.append(point)
        return torch.stack(C, dim=1)[0, :, :]

    def in_bounds(self, var, n):
        return True

    @staticmethod
    def get_timer():
        return T
