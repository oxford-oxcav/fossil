# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from torch.optim import Optimizer
from typing import Generator

import torch
from torch.optim import Optimizer

import src.learner as learner
from src.shared.cegis_values import CegisConfig
from src.shared.utils import vprint
from src.shared.consts import CertificateType


class Certificate:
    def __init__(self) -> None:
        pass

    def learn(self, optimizer: Optimizer, S: list, Sdot: list) -> dict:
        """
        param optimizer: torch optimizar
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


class Lyapunov(Certificate):
    """
    Certificies stability for CT and DT models
    bool LLO: last layer of ones in network
    XD: Symbolic formula of domain

    """

    XD = "lie-&-pos"

    def __init__(self, domains, **kw) -> None:
        self.llo = kw.get(CegisConfig.LLO.k, CegisConfig.LLO.v)
        self.domain = domains["lie-&-pos"]
        self.bias = False

    def learn(
        self, learner: learner.Learner, optimizer: Optimizer, S: list, Sdot: list
    ) -> dict:
        """
        :param learner: learner object
        :param optimizer: torch optimiser
        :param S: list of tensors of data
        :param Sdot: list of tensors containing f(data)
        :return: --
        """

        assert len(S) == len(Sdot)
        batch_size = len(S["lie-&-pos"])
        learn_loops = 1000
        margin = 0 * 0.01

        for t in range(learn_loops):
            optimizer.zero_grad()

            V, Vdot, circle = learner.forward(S["lie-&-pos"], Sdot["lie-&-pos"])

            slope = 10 ** (learner.order_of_magnitude(max(abs(Vdot)).detach()))
            leaky_relu = torch.nn.LeakyReLU(1 / slope.item())
            # compute loss function. if last layer of ones (llo), can drop parts with V
            if self.llo:
                learn_accuracy = sum(Vdot <= -margin).item()
                loss = (leaky_relu(Vdot + margin * circle)).mean()
            else:
                learn_accuracy = 0.5 * (
                    sum(Vdot <= -margin).item() + sum(V >= margin).item()
                )
                loss = (leaky_relu(Vdot + margin * circle)).mean() + (
                    leaky_relu(-V + margin * circle)
                ).mean()

            if t % 100 == 0 or t == learn_loops - 1:
                vprint(
                    (
                        t,
                        "- loss:",
                        loss.item(),
                        "- acc:",
                        learn_accuracy * 100 / batch_size,
                        "%",
                    ),
                    learner.verbose,
                )

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
        _Or = verifier.solver_fncts()["Or"]
        _And = verifier.solver_fncts()["And"]

        lyap_negated = _Or(V <= 0, Vdot > 0)
        lyap_condition = _And(self.domain, lyap_negated)
        for cs in ({"lie-&-pos": lyap_condition},):
            yield cs


class Barrier(Certificate):
    """
    Certifies Safety for CT and DT models

    Arguments:
    domains {dict}: dictionary of string:domains pairs for a initial set, unsafe set and domain

    Keyword Arguments:
    SYMMETRIC_BELT {bool}: sets belt symmetry

    """

    XD = "lie"
    XI = "init"
    XU = "unsafe"

    def __init__(self, domains, **kw) -> None:
        self.domain = domains["lie"]
        self.initial_s = domains["init"]
        self.unsafe_s = domains["unsafe"]
        self.SYMMETRIC_BELT = kw.get(
            CegisConfig.SYMMETRIC_BELT.k, CegisConfig.SYMMETRIC_BELT.v
        )
        self.bias = True

    def learn(
        self, learner: learner.Learner, optimizer: Optimizer, S: dict, Sdot: dict
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


class BarrierLyapunov(Certificate):
    """
    Certifies Safety and Stability of a model

    Arguments:
    domains {dict}: dictionary of string: domains pairs for a initial set, unsafe set and domain


    """

    def __init__(self, domains, **kw) -> None:
        self.domain = domains["lie"]
        self.initial_s = domains["init"]
        self.unsafe_s = domains["unsafe"]
        self.bias = True

    def learn(
        self, learner: learner.Learner, optimizer: Optimizer, S: list, Sdot: list
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

            # permutation_index = torch.randperm(S[0].size()[0])
            # permuted_S, permuted_Sdot = S[0][permutation_index], S_dot[0][permutation_index]
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
            percent_accuracy = percent_accuracy_init_unsafe
            slope = 1 / 10 ** 4  # (learner.orderOfMagnitude(max(abs(Vdot)).detach()))
            leaky_relu = torch.nn.LeakyReLU(slope)
            relu6 = torch.nn.ReLU6()
            # saturated_leaky_relu = torch.nn.ReLU6() - 0.01*torch.relu()
            loss = (torch.relu(B_i + margin) - slope * relu6(-B_i + margin)).mean() + (
                torch.relu(-B_u + margin) - slope * relu6(B_u + margin)
            ).mean()

            # set two belts
            percent_belt = 0

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
                        "- accuracy lie:",
                        lie_accuracy,
                    ),
                    learner.verbose,
                )

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
        _And = verifier.solver_fncts()["And"]
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
            {"init": inital_constr, "unsafe": unsafe_constr},
            {"lie": lie_constr},
        ):
            yield cs


class ReachWhileStay(Certificate):
    def __init__(self, domains, **kw) -> None:
        # TODO: Make set labels constants of the class
        self.domain = domains["lie"]
        self.initial_s = domains["init"]
        self.unsafe_s = domains["unsafe"]
        self.goal = domains["goal"]
        self.bias = True

    def learn(
        self, learner: learner.Learner, optimizer: Optimizer, S: list, Sdot: list
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


def get_certificate(certificate: CertificateType):
    if certificate == CertificateType.LYAPUNOV:
        return Lyapunov
    if certificate == CertificateType.BARRIER:
        return Barrier
    if certificate == CertificateType.BARRIER_LYAPUNOV:
        return BarrierLyapunov
    if certificate == CertificateType.RWS:
        return ReachWhileStay
