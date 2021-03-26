# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
from src.shared.component import Component


class Learner(Component):
    def __init__(self):
        super().__init__()

    def get(self, **kw):
        return self.learn(**kw)

    def learn(self, *args, **kwargs):
        return NotImplemented("Not implemented in " + self.__class__.__name__)
