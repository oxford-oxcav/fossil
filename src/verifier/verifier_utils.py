# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable

from typing import Literal

from src.verifier.verifier import Verifier
from src.verifier.drealverifier import DRealVerifier 
from src.verifier.z3verifier import Z3Verifier
from src.verifier.marabou_verifier import MarabouVerifier
from src.shared.consts import VerifierType

def get_verifier_type(verifier: Literal) -> Verifier:
    if verifier == VerifierType.DREAL:
        return DRealVerifier
    elif verifier == VerifierType.Z3:
        return Z3Verifier
    elif verifier == VerifierType.MARABOU:
        return MarabouVerifier
    else:
        raise ValueError('No verifier of type {}'.format(verifier))

def get_verifier(verifier, n_vars, constraints_method, vars_bounds, solver_vars, **kw):
    if verifier == DRealVerifier or verifier == Z3Verifier or verifier == MarabouVerifier:
        return verifier(n_vars, constraints_method, vars_bounds, solver_vars, **kw)