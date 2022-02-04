# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
from typing import Generator

import torch
from torch.optim import Optimizer

from src.certificate.certificate import Certificate
from src.shared.cegis_values import CegisConfig
from src.learner.learner import Learner
from src.shared.consts import LearningFactors
from src.shared.utils import vprint

class LyapunovCertificate(Certificate):
    """
    Certificate object for Lyapunov function synthesis
    Keyword arguments:
    bool LLO: last layer of ones in network
    XD: Symbolic formula of domain
    
    """
    XD = 'lie-&-pos'
    def __init__(self, domains, **kw) -> None:
        self.llo = kw.get(CegisConfig.LLO.k, CegisConfig.LLO.v)
        self.domain = domains['lie-&-pos']
        self.bias = False

    def learn(self, learner: Learner, optimizer: Optimizer, S: list, Sdot: list) -> dict:
        """
        :param learner: learner object
        :param optimizer: torch optimiser
        :param S: list of tensors of data
        :param Sdot: list of tensors containing f(data)
        :return: --
        """

        assert (len(S) == len(Sdot))
        batch_size = len(S['lie-&-pos'])
        learn_loops = 1000
        margin = 0*0.01

        for t in range(learn_loops):
            optimizer.zero_grad()

            V, Vdot, circle = learner.forward(S['lie-&-pos'], Sdot['lie-&-pos'])

            slope = 10 ** (learner.order_of_magnitude(max(abs(Vdot)).detach()))
            leaky_relu = torch.nn.LeakyReLU(1 / slope.item())
            # compute loss function. if last layer of ones (llo), can drop parts with V
            if self.llo:
                learn_accuracy = sum(Vdot <= -margin).item()
                loss = (leaky_relu(Vdot + margin * circle)).mean()
            else:
                learn_accuracy = 0.5 * ( sum(Vdot <= -margin).item() + sum(V >= margin).item() )
                loss = (leaky_relu(Vdot + margin * circle)).mean() + (leaky_relu(-V + margin * circle)).mean()

            if t % 100 == 0 or t == learn_loops-1:
                vprint((t, "- loss:", loss.item(), "- acc:", learn_accuracy * 100 / batch_size, '%'), learner.verbose)

            # t>=1 ensures we always have at least 1 optimisation step
            if learn_accuracy == batch_size and t >= 1:
                break

            loss.backward()
            optimizer.step()

            if learner._diagonalise:
                learner.diagonalisation()

        return {}

    def get_constraints(self, verifier, V, Vdot) -> Generator:
        """
        :param verifier: verifier object
        :param V: SMT formula of Lyapunov Function
        :param Vdot: SMT formula of Lyapunov lie derivative
        :return: tuple of dictionaries of lyapunov conditons 
        """
        _Or = verifier.solver_fncts()['Or']
        _And = verifier.solver_fncts()['And']

        lyap_negated = _Or(V <= 0, Vdot > 0)
        lyap_condition = _And(self.domain, lyap_negated)
        for cs in ({'lie-&-pos': lyap_condition}, ):
            yield cs
