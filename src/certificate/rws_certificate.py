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


class ReachWhileStayCertificate(Certificate):
    def __init__(self, domains, **kw) -> None:
        # TODO: Make set labels constants of the class
        self.domain = domains["lie"]
        self.initial_s = domains["init"]
        self.unsafe_s = domains["unsafe"]
        self.goal = domains["goal"]
        self.bias = True

    def learn(
        self, learner: Learner, optimizer: Optimizer, S: list, Sdot: list
    ) -> dict:
        """
        :param learner: learner object
        :param optimizer: torch optimiser
        :param S: list of tensors of data
        :param Sdot: list of tensors containing f(data)
        :return: --
        """
        assert len(S) == len(Sdot)

        learn_loops = 1000
        margin = 0.1
        condition_old = False
        i1 = S["lie"].shape[0]
        i2 = S["init"].shape[0]
        # I think dicts remember insertion order now, though perhaps this should be done more thoroughly
        S_cat, Sdot_cat = torch.cat((S["lie"], S["init"], S["unsafe"])), torch.cat(
            (Sdot["lie"], Sdot["init"], Sdot["unsafe"])
        )
        for t in range(learn_loops):
            optimizer.zero_grad()

            B, Bdot, _ = learner.forward(S_cat, Sdot_cat)
            B_d, Bdot_d, = (
                B[:i1],
                Bdot[:i1],
            )
            B_i = B[i1 : i1 + i2]
            B_u = B[i1 + i2 :]

            learn_accuracy = sum(B_i <= -margin).item() + sum(B_u >= margin).item()
            percent_accuracy_init_unsafe = (
                learn_accuracy * 100 / (len(S["init"]) + len(S["unsafe"]))
            )
            slope = 1 / 10 ** 4  # (learner.orderOfMagnitude(max(abs(Vdot)).detach()))
            relu6 = torch.nn.ReLU6()
            # saturated_leaky_relu = torch.nn.ReLU6() - 0.01*torch.relu()
            loss = (torch.relu(B_i + margin) - slope * relu6(-B_i + margin)).mean() + (
                torch.relu(-B_u + margin) - slope * relu6(B_u + margin)
            ).mean()

            lie_accuracy = 100 * (sum(Bdot_d <= -margin)).item() / Bdot_d.shape[0]

            loss = loss - (relu6(-Bdot_d + margin)).mean()

            # loss = loss + (100-percent_accuracy)

            if t % int(learn_loops / 10) == 0 or learn_loops - t < 10:
                vprint(
                    (
                        t,
                        "- loss:",
                        loss.item(),
                        "- accuracy init-unsafe:",
                        percent_accuracy_init_unsafe,
                        "- accuracy belt:",
                        lie_accuracy,
                    ),
                    learner.verbose,
                )

            if percent_accuracy_init_unsafe == 100 and lie_accuracy >= 99.9:
                condition = True
            else:
                condition = False

            if condition and condition_old:
                break
            condition_old = condition

            loss.backward()
            optimizer.step()

        return {}

    def get_constraints(self, verifier, C, Cdot) -> Generator:
        """
        :param verifier: verifier object
        :param C: SMT formula of Barrier function
        :param Cdot: SMT formula of Barrier lie derivative
        :return: tuple of dictionaries of Barrier conditons
        """
        _And = verifier.solver_fncts()["And"]
        _Not = verifier.solver_fncts()["Not"]
        # Cdot <= 0 in C == 0
        # C <= 0 if x \in initial
        initial_constr = _And(C > 0, self.initial_s)
        # C > 0 if x \in unsafe border
        unsafe_constr = _And(C <= 0, self.unsafe_s)

        # lie_constr = And(C >= -0.05, C <= 0.05, Cdot > 0)
        gamma = 0
        lie_constr = _And(_And(C >= 0, _Not(self.goal)), Cdot > gamma)

        # add domain constraints
        inital_constr = _And(initial_constr, self.domain)
        unsafe_constr = _And(unsafe_constr, self.domain)
        lie_constr = _And(lie_constr, self.domain)

        for cs in (
            {"init": inital_constr, "unsafe": unsafe_constr},
            {"lie": lie_constr},
        ):
            yield cs
