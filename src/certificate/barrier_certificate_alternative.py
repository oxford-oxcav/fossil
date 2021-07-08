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
from src.shared.utils import vprint

class BarrierCertificateAlternative(Certificate):
    def __init__(self, **kw) -> None:
        self.initial_s = kw.get(CegisConfig.XI.k, CegisConfig.XI.v)
        self.unsafe_s = kw.get(CegisConfig.XU.v, CegisConfig.XU.k)
        self.domain = kw.get(CegisConfig.XD.k, CegisConfig.XD.v) 

    def learn(self, learner: Learner, optimizer: Optimizer, S: list, Sdot: list) -> dict:
        """
        :param learner: learner object
        :param optimizer: torch optimiser
        :param S: list of tensors of data
        :param Sdot: list of tensors containing f(data)
        :return: --
        """
        assert (len(S) == len(Sdot))

        learn_loops = 1000
        margin = 0.1
        condition_old = False

        for t in range(learn_loops):
            optimizer.zero_grad()

            # permutation_index = torch.randperm(S[0].size()[0])
            # permuted_S, permuted_Sdot = S[0][permutation_index], S_dot[0][permutation_index]
            B_d, Bdot_d, __ = learner.numerical_net(S[0], Sdot[0])
            B_i, _, __ = learner.numerical_net(S[1], Sdot[1])
            B_u, _, __ = learner.numerical_net(S[2], Sdot[2])

            learn_accuracy = sum(B_i <= -margin).item() + sum(B_u >= margin).item()
            percent_accuracy_init_unsafe = learn_accuracy * 100 / (len(S[1]) + len(S[2]))
            percent_accuracy = percent_accuracy_init_unsafe
            slope = 1 / 10 ** 4  # (learner.orderOfMagnitude(max(abs(Vdot)).detach()))
            leaky_relu = torch.nn.LeakyReLU(slope)
            relu6 = torch.nn.ReLU6()
            # saturated_leaky_relu = torch.nn.ReLU6() - 0.01*torch.relu()
            loss = (torch.relu(B_i + margin) - slope*relu6(-B_i + margin)).mean() \
                    + (torch.relu(-B_u + margin) - slope*relu6(B_u + margin)).mean()

            # set two belts
            percent_belt = 0

            lie_accuracy =  100 * (sum(Bdot_d <= -margin)).item() /  Bdot_d.shape[0]

            loss = loss - (relu6(-Bdot_d + margin)).mean()

            # loss = loss + (100-percent_accuracy)

            if t % int(learn_loops / 10) == 0 or learn_loops - t < 10:
                vprint((t, "- loss:", loss.item(), '- accuracy init-unsafe:', percent_accuracy_init_unsafe,
                        "- accuracy lie:", lie_accuracy), learner.verbose)

            # if learn_accuracy / batch_size > 0.99:
            #     for k in range(batch_size):
            #         if Vdot[k] > -margin:
            #             print("Vdot" + str(S[k].tolist()) + " = " + str(Vdot[k].tolist()))

            if percent_accuracy_init_unsafe == 100 and percent_belt >= 99.9:
                condition = True
            else:
                condition = False

            if condition and condition_old:
                break
            condition_old = condition

            loss.backward()
            optimizer.step()

        return {}

    def get_constraints(self, verifier, B, Bdot) -> Generator:
        """
        :param verifier: verifier object
        :param B: SMT Formula of Barrier function
        :param Bdot: SMT Formula of Barrier lie derivative or one-step difference
        :return: tuple of dictionaries of Barrier conditons 
        """
        _And = verifier.solver_fncts()['And']
        # Bdot <= 0 in B == 0
        # lie_constr = And(B >= -0.05, B <= 0.05, Bdot > 0)
        lie_constr = _And(Bdot > 0)

        # B < 0 if x \in initial
        inital_constr = _And(B >= 0, self.initial_s)

        # B > 0 if x \in unsafe
        unsafe_constr = _And(B <= 0, self.unsafe_s)

        # add domain constraints
        lie_constr = _And(lie_constr, self.domain)
        inital_constr = _And(inital_constr, self.domain)
        unsafe_constr = _And(unsafe_constr, self.domain)
        for cs in (
            {'init': inital_constr, 'unsafe': unsafe_constr}, 
            {'lie': lie_constr}
            ):
            yield cs
    