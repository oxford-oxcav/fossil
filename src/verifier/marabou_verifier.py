# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# NOTE: Follow instructions for installing maraboupy, then copy
# MarabouCore.cpython-38-x86_64-linux-gnu.so from Marabou/maraboupy
# to venv/lib/python3.8/site-packages/maraboupy/

from typing import Callable, Dict
import logging
from itertools import product
from copy import deepcopy

import numpy as np

try:
    from maraboupy import Marabou
except ModuleNotFoundError:
    logging.exception("Exception while importing Marabou")


from src.verifier.verifier import Verifier
from src.shared.cegis_values import CegisConfig


class MarabouVerifier(Verifier):
    def __init__(self, n_vars, constraints_method, vars_bounds, solver_vars, **kw):
        self.inner = kw.get(CegisConfig.INNER_RADIUS.k, CegisConfig.INNER_RADIUS.v)
        self.outer = kw.get(CegisConfig.OUTER_RADIUS.k, CegisConfig.OUTER_RADIUS.v)
        super().__init__(n_vars, constraints_method, vars_bounds, solver_vars, **kw)

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
        self, V: Marabou.MarabouNetworkONNX, Vdot: Marabou.MarabouNetworkONNX
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
