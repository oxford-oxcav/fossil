# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
import torch

from src.barrier.utils import *
from src.shared.cegis_values import CegisConfig, CegisStateKeys
from src.shared.component import Component

T = Timer()


class Verifier(Component):
    def __init__(self, n_vars, whole_domain, initial_state, unsafe_state, vars_bounds, solver_vars, **kw):
        super().__init__()
        self.iter = -1
        self.n = n_vars
        self.domain = whole_domain
        self.initial_s = initial_state
        self.counterexample_n = 20
        self.unsafe_s = unsafe_state
        self._last_cex = []
        self._n_cex_to_keep = self.counterexample_n * 1
        self.xs = solver_vars
        self._solver_timeout = 30
        self._vars_bounds = vars_bounds
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
        # translator default returns V and Vdot
        return self.verify(kw[CegisStateKeys.V], kw[CegisStateKeys.V_dot])

    @timer(T)
    def verify(self, B, Bdot):
        """
        :param V: z3 expr
        :param Vdot: z3 expr
        :return:
                found_lyap: True if V is valid
                C: a list of ctx
        """
        found = False
        s_init = self.new_solver()
        s_unsafe = self.new_solver()
        s_lie = self.new_solver()
        # the order of elements is: domain, initial, unsafe
        solvers = {'lie': s_lie, 'init': s_init, 'unsafe': s_unsafe}
        fmls = self.domain_constraints(B, Bdot)

        # if sat, found counterexample; if unsat, V is lyap
        res_init, timedout = self.solve_with_timeout(s_init, fmls['init'])
        if timedout:
            vprint("init timed out", self.verbose)

        res_unsafe, timedout = self.solve_with_timeout(s_unsafe, fmls['unsafe'])
        if timedout:
            vprint("unsafe timed out", self.verbose)

        res_lie = []
        no_cex_init = self.is_unsat(res_init)
        no_cex_unsafe = self.is_unsat(res_unsafe)
        if no_cex_init and no_cex_unsafe:
            self._solver_timeout = 150
            res_lie, timedout = self.solve_with_timeout(s_lie, fmls['lie'])
            if timedout:
                vprint("timed out -> try smooth Lie", self.verbose)
                self._solver_timeout = 180
                res_lie, s_lie = self.smoothed_lie(B, Bdot)
                solvers = {'lie': s_lie, 'init': s_init, 'unsafe': s_unsafe}
                if timedout:
                    vprint(":/ timed out -> try fail_safe", self.verbose)
                    res_lie, s_lie = self.fail_safe(B, Bdot)
                    solvers = {'lie': s_lie, 'init': s_init, 'unsafe': s_unsafe}

        ces = [[], [], []]

        results = {'lie': res_lie, 'init': res_init, 'unsafe': res_unsafe}
        if all(self.is_unsat(res) for res in results.values()):
            vprint(['No counterexamples found!'], self.verbose)
            found = True
        else:
            for index, o in enumerate(results.items()):
                solver, res = o
                if self.is_sat(res):
                    original_point = self.compute_model(solvers[solver], res)
                    ces[index] = self.randomise_counterex(original_point)
                else:
                    vprint([res], self.verbose)

        return {CegisStateKeys.found: found, CegisStateKeys.cex: ces}

    def normalize_number(self, n):
        return n

    def domain_constraints(self, B, Bdot):
        _And = self.solver_fncts()['And']
        # Bdot <= 0 in B == 0
        # lie_constr = And(B >= -0.05, B <= 0.05, Bdot > 0)
        lie_constr = _And(B == 0, Bdot >= 0)

        # B < 0 if x \in initial
        inital_constr = _And(B >= 0, self.initial_s)

        # B > 0 if x \in unsafe
        unsafe_constr = _And(B <= 0, self.unsafe_s)

        # add domain constraints
        lie_constr = _And(lie_constr, self.domain)
        inital_constr = _And(inital_constr, self.domain)
        unsafe_constr = _And(unsafe_constr, self.domain)
        return {
            'lie': lie_constr,
            'init': inital_constr,
            'unsafe': unsafe_constr,
        }

    def circle_constr(self, c, r):
        """
        :param x:
        :param c:
        :return:
        """
        circle_constr = np.sum([(x - c[i]) ** 2 for i, x in enumerate(self.xs)]) <= r

        return circle_constr

    def square_constr(self, domain):
        """
        :param domain:
        :return:
        """
        square_constr = []
        for idx, x in enumerate(self.xs):
            try:
                square_constr += [x[0, 0] >= domain[idx][0]]
                square_constr += [x[0, 0] <= domain[idx][1]]
            except:
                square_constr += [x >= domain[idx][0]]
                square_constr += [x <= domain[idx][1]]
        return square_constr

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
            n = self._model_result(solver, model, x, i)
            normalized = self.normalize_number(n)
            temp += [normalized]

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
            random_point = point + 5*1e-4 * torch.randn(shape)
            # if self.inner < torch.norm(random_point) < self.outer:
            C.append(random_point)
        C.append(point)
        return torch.stack(C, dim=1)[0, :, :]

    def smoothed_lie(self, B, Bdot):
        """
        :param B:
        :return:
        """
        _And = self.solver_fncts()['And']
        s = self.new_solver()
        f = _And(B >= -0.05, B <= 0.05, Bdot >= 0)
        f = _And(f, self.domain)
        res_smooth, timedout = self.solve_with_timeout(s, f)

        return res_smooth, s

    def fail_safe(self, B, Bdot):
        """
        :param B:
        :return:
        """
        _And = self.solver_fncts()['And']
        s = self.new_solver()
        f = _And(_And(B >= 0.0, Bdot >= 0))
        f = _And(f, self.domain)
        res_zero, timedout = self.solve_with_timeout(s, f)

        if timedout:
            print('fail_safe timedout')

        return res_zero, s

    def in_bounds(self, var, n):
        left, right = self._vars_bounds[var]
        return left < n < right

    @staticmethod
    def get_timer():
        return T
