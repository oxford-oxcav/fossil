# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import timeit
from copy import deepcopy
from itertools import product
from typing import Any, Callable, Dict, Literal, Union

import numpy as np
import torch
import z3
from cvc5 import pythonic as cvpy

try:
    import dreal as dr
except ModuleNotFoundError:
    logging.exception("No dreal")

import fossil.logger as logger
from fossil.component import Component
from fossil.consts import *
from fossil.utils import (
    Timer,
    contains_object,
    dreal_replacements,
    timer,
    z3_replacements,
)

T = Timer()
SYMBOL = Union[z3.ArithRef, dr.Variable, dr.Expression]
ver_log = logger.Logger.setup_logger(__name__)


def optional_Marabou_import():
    """Function to import Marabou only if needed."""
    try:
        from maraboupy import Marabou
    except ImportError as e:
        ver_log.exception("Exception while importing Marabou")


@dataclass
class VerifierConfig:
    DREAL_INFINITY: float = 1e300
    # dReal will return infinity when:
    # - no counterexamples are found
    # - a smaller counterexample also exists
    # check again for a counterexample with the bound below
    DREAL_SECOND_CHANCE_BOUND: float = 1e3
    DELTA: float = 1e-4
    VARS_BOUNDS: tuple = (-DREAL_INFINITY, DREAL_INFINITY)

    def __getitem__(self, item):
        return getattr(self, item)


class Verifier(Component):
    def __init__(self, n_vars, constraints_method, solver_vars, verbose):
        super().__init__()
        self.iter = -1
        self.n = n_vars
        self.counterexample_n = 20
        self._last_cex = []
        self._n_cex_to_keep = self.counterexample_n * 1
        self.xs = solver_vars
        self._solver_timeout = 30
        self.constraints_method = constraints_method
        self.verbose = verbose
        self.optional_configs = VerifierConfig()
        self._vars_bounds = [self.optional_configs.VARS_BOUNDS for _ in range(n_vars)]
        assert self.counterexample_n > 0

    @staticmethod
    def new_vars(n):
        """Example: return [Real('x%d' % i) for i in range(n_vars)]"""
        raise NotImplementedError("")

    @staticmethod
    def solver_fncts():
        """Example: return {'And': z3.And}"""
        raise NotImplementedError("")

    def new_solver(self):
        """Example: return z3.Solver()"""
        raise NotImplementedError("")

    def is_sat(self, res) -> bool:
        """Example: return res == sat"""
        raise NotImplementedError("")

    def is_unsat(self, res) -> bool:
        """Example: return res == unsat"""
        raise NotImplementedError("")

    def _solver_solve(self, solver, fml):
        """Example: solver.add(fml); return solver.check()"""
        raise NotImplementedError("")

    def _solver_model(self, solver, res):
        """Example: return solver.model()"""
        raise NotImplementedError("")

    def _model_result(self, solver, model, var, idx):
        """Example: return float(model[var[0, 0]].as_fraction())"""
        raise NotImplementedError("")

    def get(self, **kw):
        # translator default returns V and Vdot
        return self.verify(kw[CegisStateKeys.V], kw[CegisStateKeys.V_dot])

    def verify(self, C, dC):
        """
        :param C: z3 expr
        :param dC: z3 expr
        :return:
                found_lyap: True if C is valid
                C: a list of ctx
        """
        found, timed_out = False, False
        fmls = self.domain_constraints(C, dC)
        results = {}
        solvers = {}

        for group in fmls:
            for label, condition in group.items():
                s = self.new_solver()
                res, timedout = self.solve_with_timeout(s, condition)
                results[label] = res
                solvers[label] = s
                # if sat, found counterexample; if unsat, C is lyap
                if timedout:
                    ver_log.info(label + "timed out")
            if any(self.is_sat(res) for res in results.values()):
                break

        ces = {label: [] for label in results.keys()}  # [[] for res in results.keys()]

        if all(self.is_unsat(res) for res in results.values()):
            ver_log.info("No counterexamples found!")
            found = True
        else:
            for index, o in enumerate(results.items()):
                label, res = o
                if self.is_sat(res):
                    ver_log.info(label + ": ")
                    original_point = self.compute_model(solvers[label], res)
                    # This next bit is hard to handle if C, dC are tuples of formula
                    # For now, just check it's not a tuple
                    if type(C) is not tuple:
                        V_ctx, Vdot_ctx = self.replace_point(
                            C, self.xs, original_point.numpy().T
                        ), self.replace_point(dC, self.xs, original_point.numpy().T)
                        ver_log.info("\nV_ctx: {} ".format(V_ctx))
                        ver_log.info("\nVdot_ctx: {} ".format(Vdot_ctx))
                    ces[label] = self.randomise_counterex(original_point)
                else:
                    ver_log.info(res)

        return {CegisStateKeys.found: found, CegisStateKeys.cex: ces}

    def normalize_number(self, n):
        return n

    def domain_constraints(self, C, dC):
        return self.constraints_method(self, C, dC)

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

    @timer(T)
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
        ver_log.debug("Fml: {}".format(fml))
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
        ver_log.info("Counterexample Found: {}".format(model))
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
            random_point = point + 5 * 1e-4 * torch.randn(shape)
            # if self.inner < torch.norm(random_point) < self.outer:
            C.append(random_point)
        C.append(point)
        return torch.stack(C, dim=1)[0, :, :]

    def smoothed_lie(self, B, Bdot):
        """
        :param B:
        :return:
        """
        _And = self.solver_fncts()["And"]
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
        _And = self.solver_fncts()["And"]
        s = self.new_solver()
        f = _And(_And(B >= 0.0, Bdot >= 0))
        f = _And(f, self.domain)
        res_zero, timedout = self.solve_with_timeout(s, f)

        if timedout:
            print("fail_safe timedout")

        return res_zero, s

    def in_bounds(self, var, n):
        left, right = self._vars_bounds[var]
        return left < n < right

    @staticmethod
    def get_timer():
        return T


class VerifierDReal(Verifier):
    @staticmethod
    def new_vars(n, base="x"):
        return [dr.Variable(base + str(i)) for i in range(n)]

    def new_solver(self):
        return None

    @staticmethod
    def check_type(x) -> bool:
        """
        :param x: any
        :returns: True if Dreal compatible, else false
        """
        return contains_object(x, dr.Variable)

    @staticmethod
    def solver_fncts() -> Dict[str, Callable]:
        return {
            "sin": dr.sin,
            "cos": dr.cos,
            "exp": dr.exp,
            "And": dr.And,
            "Or": dr.Or,
            "If": dr.if_then_else,
            "Check": VerifierDReal.check_type,
            "Not": dr.Not,
            "False": dr.Formula.FALSE(),
            "True": dr.Formula.TRUE(),
        }

    def is_sat(self, res) -> bool:
        return isinstance(res, dr.Box)

    @staticmethod
    def replace_point(expr, ver_vars, point):
        return dreal_replacements(expr, ver_vars, point)

    def is_unsat(self, res) -> bool:
        # int(str("x0")) = 0
        bounds_not_ok = not self.within_bounds(res)
        return res is None or bounds_not_ok

    def within_bounds(self, res) -> bool:
        return isinstance(res, dr.Box) and all(
            self.in_bounds(int(str(x)[1:]), interval.mid())
            for x, interval in res.items()
        )

    def _solver_solve(self, solver, fml):
        res = dr.CheckSatisfiability(fml, 0.0001)
        if self.is_sat(res) and not self.within_bounds(res):
            ver_log.info("Second chance bound used")
            new_bound = self.optional_configs.DREAL_SECOND_CHANCE_BOUND
            fml = dr.And(fml, *(dr.And(x < new_bound, x > -new_bound) for x in self.xs))
            res = dr.CheckSatisfiability(fml, 0.0001)
        return res

    def _solver_model(self, solver, res):
        assert self.is_sat(res)
        return res

    def _model_result(self, solver, model, x, idx):
        return float(model[idx].mid())


class VerifierZ3(Verifier):
    @staticmethod
    def new_vars(n, base="x"):
        return [z3.Real(base + str(i)) for i in range(n)]

    def new_solver(self):
        return z3.Solver()

    @staticmethod
    def check_type(x) -> bool:
        """
        :param x: any
        :returns: True if z3 compatible, else false
        """
        return contains_object(x, z3.ArithRef)

    @staticmethod
    def solver_fncts() -> Dict[str, Callable]:
        return {
            "And": z3.And,
            "Or": z3.Or,
            "If": z3.If,
            "Check": VerifierZ3.check_type,
            "Not": z3.Not,
            "False": False,
            "True": True,
        }

    @staticmethod
    def replace_point(expr, ver_vars, point):
        return z3_replacements(expr, ver_vars, point)

    def is_sat(self, res) -> bool:
        return res == z3.sat

    def is_unsat(self, res) -> bool:
        return res == z3.unsat

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
        except TypeError:
            try:
                return float(model[x[0, 0]].as_fraction())
            except:  # when z3 finds non-rational numbers, prints them w/ '?' at the end --> approx 10 decimals
                return float(model[x[0, 0]].approx(10).as_fraction())


class VerifierCVC5(Verifier):
    @staticmethod
    def new_vars(n, base="x"):
        return [cvpy.Real(base + str(i)) for i in range(n)]

    def new_solver(self):
        s = cvpy.Solver()
        # set logic to QF_NRA. This step seems unnecessary but also seems to massively speed up cvc5.
        # I think there must be better ways to interface with cvc5 but sticking with the z3 interface is most convenient for now
        s.solver.setLogic("NRA")
        # s.solver.setOption("nl-ext", "full")
        # s.solver.setOption("nl-cov", "false")
        return s

    @staticmethod
    def check_type(x) -> bool:
        """
        :param x: any
        :returns: True if cvc5 compatible, else false
        """
        return contains_object(x, cvpy.ArithRef)

    @staticmethod
    def solver_fncts() -> Dict[str, Callable]:
        return {
            "And": cvpy.And,
            "Or": cvpy.Or,
            "If": cvpy.If,
            "Check": VerifierCVC5.check_type,
            "Not": cvpy.Not,
            "False": False,
            "True": True,
        }

    @staticmethod
    def replace_point(expr, ver_vars, point):
        """
        :param expr: cvc expr
        :param cvc_vars: cvc vars, matrix
        :param ctx: matrix of numerical values
        :return: value of V, Vdot in ctx
        """
        replacements = []
        for i in range(len(ver_vars)):
            try:
                replacements += [(ver_vars[i, 0], cvpy.RealVal(point[i, 0]))]
            except TypeError:
                replacements += [(ver_vars[i], cvpy.RealVal(point[i, 0]))]

        replaced = cvpy.substitute(expr, replacements)

        return cvpy.simplify(replaced)

    def is_sat(self, res) -> bool:
        return res == cvpy.sat

    def is_unsat(self, res) -> bool:
        return res == cvpy.unsat

    def _solver_solve(self, solver, fml):
        solver.add(cvpy.simplify(fml))
        return solver.check()

    def _solver_model(self, solver, res):
        return solver.model()

    def _model_result(self, solver, model, x, i):
        try:
            return float(model[x].as_fraction())
        except AttributeError:
            return float(model[x].approx(10).as_fraction())
        except TypeError:
            try:
                return float(model[x[0, 0]].as_fraction())
            except:  # when z3 finds non-rational numbers, prints them w/ '?' at the end --> approx 10 decimals
                return float(model[x[0, 0]].approx(10).as_fraction())


class VerifierMarabou(Verifier):
    def __init__(self, n_vars, constraints_method, vars_bounds, solver_vars, **kw):
        self.inner = kw.get(CegisConfig.INNER_RADIUS.k, CegisConfig.INNER_RADIUS.v)
        self.outer = kw.get(CegisConfig.OUTER_RADIUS.k, CegisConfig.OUTER_RADIUS.v)
        super().__init__(n_vars, constraints_method, vars_bounds, solver_vars, **kw)
        optional_Marabou_import()

    @staticmethod
    def new_vars(n):
        return range(n)

    def new_solver(self):
        return Marabou.createOptions(verbosity=0)

    @staticmethod
    def solver_fncts() -> Dict[str, Callable]:
        return {"And": None, "Or": None, "Estim_Activation": np.max}

    def is_sat(self, res: Dict) -> bool:
        return bool(res)

    def is_unsat(self, res: Dict) -> bool:
        return not bool(res)

    def _solver_solve(self, solver, fml):
        results = [
            quadrant.solve(options=solver, verbose=False)[0] for quadrant in fml
        ]  # List of dicts of counter examples for each quadrant
        combined_results = {}
        for result in results:
            combined_results.update(
                result
            )  # This means counterexamples are lost - not a permanent solution
        return combined_results

    def _solver_model(self, solver, res):
        return res

    def _model_result(self, solver, model, x, i):
        return model[x]

    def domain_constraints(
        self, V: "Marabou.MarabouNetworkONNX", Vdot: "Marabou.MarabouNetworkONNX"
    ):
        """
        :param V:
        :param Vdot:
        :return: tuple of Marabou ONNX networks with appropriately set input/ output bounds
        """
        inner = self.inner
        outer = self.outer

        orthants = list(product(*[[1.0, -1.0] for i in range(self.n)]))
        V_input_vars = V.inputVars[0][0]
        Vdot_input_vars = Vdot.inputVars[0][0]
        V_output_vars = V.outputVars[0]
        Vdot_output_vars = Vdot.outputVars[0]

        V_tuple = tuple(deepcopy(V) for i in range(len(orthants)))
        Vdot_tuple = tuple(
            deepcopy(Vdot) for i in range(len(orthants))
        )  # Definitely bad usage of memory - a generator would be better
        assert (V_input_vars == Vdot_input_vars).all()

        # TODO: This now covers all 2^n orthants, but excludes a small 'cross' region around the axis - I think this is why
        # your approach is to add regions that overlap. Will add this but not urgent just yet (seems to work for very low inner cords)
        for i, orthant in enumerate(orthants):
            for j, var in enumerate(V_input_vars):
                V_tuple[i].setLowerBound(
                    var, min(orthant[j] * inner, orthant[j] * outer)
                )
                V_tuple[i].setUpperBound(
                    var, max(orthant[j] * inner, orthant[j] * outer)
                )
                Vdot_tuple[i].setLowerBound(
                    var, min(orthant[j] * inner, orthant[j] * outer)
                )
                Vdot_tuple[i].setUpperBound(
                    var, max(orthant[j] * inner, orthant[j] * outer)
                )

            V_tuple[i].setUpperBound(V_output_vars[0], 0)
            Vdot_tuple[i].setLowerBound(Vdot_output_vars[0], 0)

        pass

        for cs in ({"lyap": (*V_tuple, *Vdot_tuple)},):
            yield cs


def get_verifier_type(verifier: Literal) -> Verifier:
    if verifier == VerifierType.DREAL:
        return VerifierDReal
    elif verifier == VerifierType.Z3:
        return VerifierZ3
    elif verifier == VerifierType.CVC5:
        return VerifierCVC5
    elif verifier == VerifierType.MARABOU:
        return VerifierMarabou
    else:
        raise ValueError("No verifier of type {}".format(verifier))


def get_verifier(verifier, n_vars, constraints_method, solver_vars, verbose):
    if (
        verifier == VerifierDReal
        or verifier == VerifierZ3
        or verifier == VerifierCVC5
        or verifier == VerifierMarabou
    ):
        return verifier(n_vars, constraints_method, solver_vars, verbose)
    else:
        raise ValueError("No verifier of type {}".format(verifier))
