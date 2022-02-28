# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sympy as sp

from src.learner.learner import Learner


class SympySolver(Learner):
    @staticmethod
    def solver_fncts() -> {}:
        return {
            "And": sp.And,
            "Or": sp.Or,
            "If": sp.ITE,
            "sin": sp.sin,
            "cos": sp.cos,
            "exp": sp.exp,
        }
