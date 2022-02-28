# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable

from typing import Literal

from src.learner.learner import Learner
from src.learner.net_continuous import NNContinuous
from src.learner.net_discrete import NNDiscrete
from src.shared.consts import TimeDomain

def get_learner(time_domain: Literal) -> Learner:
    if time_domain == TimeDomain.CONTINUOUS:
        return NNContinuous
    elif time_domain == TimeDomain.DISCRETE:
        return NNDiscrete
    else:
        raise ValueError('Learner not implemented')