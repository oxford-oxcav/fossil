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


class BarrierCertificate(Certificate):
    """
    Certifies Safety for CT and DT models

    Arguments:
    domains {dict}: dictionary of string:domains pairs for a initial set, unsafe set and domain
    
    Keyword Arguments:
    SYMMETRIC_BELT {bool}: sets belt symmetry

    """

    def __init__(self, domains, **kw) -> None:
        self.domain = domains["lie"]
        self.initial_s = domains["init"]
        self.unsafe_s = domains["unsafe"]
        self.SYMMETRIC_BELT = kw.get(
            CegisConfig.SYMMETRIC_BELT.k, CegisConfig.SYMMETRIC_BELT.v
        )
        self.bias = True

    def learn(
        self, learner: Learner, optimizer: Optimizer, S: dict, Sdot: dict
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
        S_cat, Sdot_cat = torch.cat([s for s in S.values()]), torch.cat(
            [sdot for sdot in Sdot.values()]
        )

        for t in range(learn_loops):
            optimizer.zero_grad()

            # This seems slightly faster
            B, Bdot, _ = learner.forward(S_cat, Sdot_cat)
            B_d, Bdot_d, = (
                B[:i1],
                Bdot[:i1],
            )
            B_i = B[i1 : i1 + i2]
            B_u = B[i1 + i2 :]

            learn_accuracy = sum(B_i <= -margin).item() + sum(B_u >= margin).item()
            percent_accuracy_init_unsafe = (
                learn_accuracy * 100 / (len(S["unsafe"]) + len(S["init"]))
            )
            slope = 1 / 10 ** 4
            relu6 = torch.nn.ReLU6()
            loss = (torch.relu(B_i + margin) - slope * relu6(-B_i + margin)).mean() + (
                torch.relu(-B_u + margin) - slope * relu6(B_u + margin)
            ).mean()

            # set two belts
            percent_belt = 0
            if self.SYMMETRIC_BELT:
                belt_index = torch.nonzero(torch.abs(B_d) <= 0.5)
            else:
                belt_index = torch.nonzero(B_d >= -margin)

            if belt_index.nelement() != 0:
                dB_belt = torch.index_select(Bdot_d, dim=0, index=belt_index[:, 0])
                learn_accuracy = learn_accuracy + (sum(dB_belt <= -margin)).item()
                percent_belt = 100 * (sum(dB_belt <= -margin)).item() / dB_belt.shape[0]

                loss = loss - (relu6(-dB_belt + 0 * margin)).mean()

            if t % int(learn_loops / 10) == 0 or learn_loops - t < 10:
                vprint(
                    (
                        t,
                        "- loss:",
                        loss.item(),
                        "- accuracy init-unsafe:",
                        percent_accuracy_init_unsafe,
                        "- accuracy belt:",
                        percent_belt,
                        "- points in belt:",
                        len(belt_index),
                    ),
                    learner.verbose,
                )

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
        :param B: SMT formula of Barrier function
        :param Bdot: SMT formula of Barrier lie derivative
        :return: tuple of dictionaries of Barrier conditons
        """
        _And = verifier.solver_fncts()["And"]
        _Or = verifier.solver_fncts()["Or"]
        _Not = verifier.solver_fncts()["Not"]
        # Bdot <= 0 in B == 0
        # lie_constr = And(B >= -0.05, B <= 0.05, Bdot > 0)
        # lie_constr = _Not(_Or(Bdot < 0, _Not(B==0)))
        lie_constr = _And(B == 0, Bdot >= 0)

        # B < 0 if x \in initial
        initial_constr = _And(B >= 0, self.initial_s)

        # B > 0 if x \in unsafe
        unsafe_constr = _And(B <= 0, self.unsafe_s)

        # add domain constraints
        lie_constr = _And(lie_constr, self.domain)
        inital_constr = _And(initial_constr, self.domain)
        unsafe_constr = _And(unsafe_constr, self.domain)

        for cs in (
            {"init": inital_constr, "unsafe": unsafe_constr},
            {"lie": lie_constr},
        ):
            yield cs
