# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable

from typing import Literal

from src.shared.component import Component
from src.translator.translator_continuous import TranslatorContinuous
from src.translator.translator_discrete import TranslatorDiscrete
from src.translator.translator_marabou import MarabouTranslator
from src.shared.consts import VerifierType, TimeDomain


def get_translator_type(time_domain: Literal, verifier: Literal) -> Component:
    if verifier == VerifierType.MARABOU:
        if time_domain != TimeDomain.DISCRETE:
            raise ValueError(
                "Marabou verifier not compatible with continuous-time dynamics"
            )
        return MarabouTranslator
    elif time_domain == TimeDomain.DISCRETE:
        return TranslatorDiscrete
    elif time_domain == TimeDomain.CONTINUOUS:
        return TranslatorContinuous
    else:
        TypeError("Not Implemented Translator")


def get_translator(translator_type: Component, net, x, xdot, eq, rounding, **kw):
    if translator_type == TranslatorContinuous or translator_type == TranslatorDiscrete:
        return translator_type(net, x, xdot, eq, rounding, **kw)
    elif translator_type == MarabouTranslator:
        return translator_type(x.shape[0])
