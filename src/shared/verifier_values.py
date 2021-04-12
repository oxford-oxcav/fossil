# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
from aenum import Enum, NoAlias


class VerifierConfig(Enum, settings=NoAlias):
    DREAL_INFINITY              = 1e300
    # dReal will return infinity when:
    # - no counterexamples are found
    # - a smaller counterexample also exists
    # check again for a counterexample with the bound below
    DREAL_SECOND_CHANCE_BOUND   = 1e3

    @property
    def k(self):
        return self.name

    @property
    def v(self):
        return self.value
