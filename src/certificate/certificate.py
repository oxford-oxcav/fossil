# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 

from torch.optim import Optimizer

class Certificate:

    def __init__(self) -> None:
        pass 

    def learn(self, optimizer: Optimizer, S: list, Sdot: list) -> dict:
        """
        param optimizer: torch optimizar
        param S:
        param S:
        """
        raise NotImplemented("Not implemented in " + self.__class__.__name__)

    def get_constraints(self, C, Cdot) -> tuple:
        """
        param C: SMT Formula of Certificate
        param Cdot: SMT Formula of Certificate time derivative or one-step difference
        return: tuple of dictionaries of certificate conditons 
        """
        raise NotImplemented("Not implemented in " + self.__class__.__name__)






