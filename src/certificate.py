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
    SD = XD

    def __init__(self, domains, **kw) -> None:
        self.llo = kw.get(CegisConfig.LLO.k, CegisConfig.LLO.v)
        self.domain = domains[Lyapunov.XD]
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
        batch_size = len(S[Lyapunov.SD])
        learn_loops = 1000
        margin = 0 * 0.01

        for t in range(learn_loops):
            optimizer.zero_grad()

            V, Vdot, circle = learner.forward(S[Lyapunov.SD], Sdot[Lyapunov.SD])

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
        for cs in ({Lyapunov.SD: lyap_condition},):
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
    SD = XD
    SI = XI
    SU = XU

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

                loss = (
                    loss
                    + (relu6(dB_belt + 0 * margin)).mean()
                    - slope * relu6(-dB_belt).mean()
                )

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
            {Barrier.SI: inital_constr, Barrier.SU: unsafe_constr},
            {Barrier.SD: lie_constr},
        ):
            yield cs


class BarrierLyapunov(Certificate):
    """
    Certifies Safety of a model

    Arguments:
    domains {dict}: dictionary of string: domains pairs for a initial set, unsafe set and domain


    """

    XD = "lie"
    XI = "init"
    XU = "unsafe"
    SD = XD
    SI = XI
    SU = XU

    def __init__(self, domains, **kw) -> None:
        self.domain = domains[BarrierLyapunov.XD]
        self.initial_s = domains[BarrierLyapunov.XI]
        self.unsafe_s = domains[BarrierLyapunov.XU]
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
        i1 = S[BarrierLyapunov.XD].shape[0]
        i2 = S[BarrierLyapunov.SD].shape[0]
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
                learn_accuracy
                * 100
                / (len(S[BarrierLyapunov.SU]) + len(S[BarrierLyapunov.SD]))
            )
            slope = 1 / 10 ** 4  # (learner.orderOfMagnitude(max(abs(Vdot)).detach()))
            relu6 = torch.nn.ReLU6()
            p = 1
            init_loss = (torch.relu(B_i + margin) - slope * relu6(-B_i + margin)).mean()
            # init_loss = (init_loss / init_loss.detach().norm(p=p)).mean()
            unsafe_loss = (
                torch.relu(-B_u + margin) - slope * relu6(B_u + margin)
            ).mean()
            # unsafe_loss = (unsafe_loss / unsafe_loss.detach().norm(p=p)).mean()

            lie_loss = (relu6(Bdot_d + margin)).mean()
            # lie_loss = (lie_loss / lie_loss.detach().norm(p=p)).mean()

            # set two belts
            percent_belt = 0

            lie_accuracy = 100 * (sum(Bdot_d <= -margin)).item() / Bdot_d.shape[0]

            loss = init_loss + unsafe_loss + lie_loss

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
            {BarrierLyapunov.SD: inital_constr, BarrierLyapunov.SU: unsafe_constr},
            {BarrierLyapunov.SD: lie_constr},
        ):
            yield cs


class RWS(Certificate):
    XD = "lie"
    XI = "init"
    XS = "safe"
    dXS = "safe_border"
    XG = "goal"
    SD = XD
    SI = XI
    SS = XS
    SG = XG

    # Reach While stay must satisfy:
    # \forall x in XI, V <= 0,
    # \forall x in boundary of XS, V > 0,
    # \forall x in A \ XG, dV/dt <= 0
    # A = {x \in XS| V <=0 }

    def __init__(self, domains, **kw) -> None:
        # TODO: Make set labels constants of the class
        self.domain = domains[RWS.XD]
        self.initial_s = domains[RWS.XI]
        self.safe_s = domains[RWS.XS]
        self.safe_border = domains[RWS.dXS]
        self.goal = domains[RWS.XG]
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
        i1 = S[RWS.XD].shape[0]
        i2 = S[RWS.XI].shape[0]
        # I think dicts remember insertion order now, though perhaps this should be done more thoroughly
        S_cat, Sdot_cat = torch.cat(
            (S[RWS.XD], S[RWS.XI], S[RWS.XS])
        ), torch.cat(
            (Sdot[RWS.XD], Sdot[RWS.XI], Sdot[RWS.XS])
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
                learn_accuracy
                * 100
                / (len(S[RWS.XI]) + len(S[RWS.XS]))
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
        # C > 0 if x \in safe border
        safe_constr = _And(C <= 0, self.safe_border)

        # lie_constr = And(C >= -0.05, C <= 0.05, Cdot > 0)
        gamma = 0
        lie_constr = _And(_And(C >= 0, _Not(self.goal)), Cdot > gamma)

        # add domain constraints
        inital_constr = _And(initial_constr, self.domain)
        safe_constr = _And(safe_constr, self.domain)
        lie_constr = _And(lie_constr, self.domain)

        
        for cs in (
            {RWS.XI: inital_constr, RWS.XS: safe_constr},
            {RWS.XD: lie_constr},
        ):
            yield cs

class RSWS(Certificate):
    """ Reach and Stay While Stay Certificate

    Firstly satisfies reach while stay conditions, given by:
        forall x in XI, V <= 0,
        forall x in boundary of XS, V > 0,
        forall x in A \ XG, dV/dt <= 0
        A = {x \in XS| V <=0 }

    http://arxiv.org/abs/1812.02711
    In addition to the RWS properties, to satisfy RSWS:
    forall x in border XG: V > \beta
    forall x in XG \ int(B): dV/dt <= 0
    B = {x in XS | V <= \beta}
    Best to ask SMT solver if a beta exists such that the above holds -
    but currently we don't train for this condition.

    Crucially, this relies only on the border of the safe set, 
    rather than the safe set itself.
    Since the border must be positive (and the safe invariant negative), this is inline
    with making the complement (the unsafe set) positive. Therefore, the implementation 
    requires an unsafe set to be passed in, and assumes its border is the same of the border of the safe set.
    """

    XD = "lie"
    XI = "init"
    XU = "unsafe_border"
    XS = "safe"
    XG = "goal"
    dXG = "goal_border"
    SD = XD
    SI = XI
    SU = "unsafe"
    SG = XG

    def __init__(self, domains, **kw) -> None:
        """initialise the RSWS certificate

        Args:
            domains (Dict): contains symbolic formula of the domains
        """
        self.domain = domains[RSWS.XD]
        self.initial_s = domains[RSWS.XI]
        self.unsafe_border = domains[RSWS.XU]
        self.safe = domains[RSWS.XS]
        self.goal = domains[RSWS.XG]
        self.goal_border = domains[RSWS.dXG]
        self.bias = True

    def learn(
        self, learner: learner.Learner, optimizer: Optimizer, S: list, Sdot: list
    ) -> dict:
        """_summary_

        Args:
            learner (learner.Learner): learner object
            optimizer (Optimizer): torch optimizer 
            S (list): list of tensors of the state
            Sdot (list): list of tensors of the state derivative

        Returns:
            dict: empty dict
        """    
        assert len(S) == len(Sdot)

        learn_loops = 1000
        margin = 0.1
        condition_old = False
        i1 = S[RSWS.XD].shape[0]
        i2 = S[RSWS.XI].shape[0]
        S_cat, Sdot_cat = torch.cat(
            (S[RSWS.XD], S[RSWS.XI], S[RSWS.SU])
        ), torch.cat(
            (Sdot[RSWS.XD], Sdot[RSWS.XI], Sdot[RSWS.SU])
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
                learn_accuracy
                * 100
                / (len(S[RSWS.XI]) + len(S[RSWS.SU]))
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
                        "- accuracy init-safe:",
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
        """returns the constraints of the certificate

        Args:
            verifier (Verifier): verifier object
            C: SMT formula of certificate function
            Cdot: SMT formula of certificate lie derivative

        Yields:
            Generator: tuple of dictionaries of certificate conditons
        """        
        _And = verifier.solver_fncts()["And"]
        _Not = verifier.solver_fncts()["Not"]
        # Cdot <= 0 in C == 0
        # C <= 0 if x \in initial
        initial_constr = _And(C > 0, self.initial_s)
        # C > 0 if x \in safe border
        safe_constr = _And(C <= 0, self.unsafe_border)

        # lie_constr = And(C >= -0.05, C <= 0.05, Cdot > 0)
        gamma = 0
        lie_constr = _And(_And(C >= 0, _Not(self.goal)), Cdot > gamma)

        # add domain constraints
        inital_constr = _And(initial_constr, self.domain)
        safe_constr = _And(safe_constr, self.domain)
        lie_constr = _And(lie_constr, self.domain)

        for cs in (
            {RSWS.XI: inital_constr, RSWS.XU: safe_constr},
            {RSWS.XD: lie_constr},
        ):
            yield cs
    
    def stay_in_goal_check(self, verifier, C, Cdot):
        """Checks if the system stays in the goal region. True if it stays in the goal region, False otherwise.

        This check involves finding a beta such that:

        \forall x in border XG: V > \beta
        \forall x in XG \ int(B): dV/dt <= 0
        B = {x in XS | V <= \beta}

        Args:
            verifier (Verifier): verifier object
            C: SMT formula of certificate function
            Cdot: SMT formula of certificate lie derivative

        Returns:
            bool: True if sat
        """
        _And = verifier.solver_fncts()["And"]
        _Not = verifier.solver_fncts()["Not"]
        beta = verifier.new_vars(1, base='b')[0]
        B = _And(C <= beta, _And(self.domain, self.safe)) 
        dG = self.goal_border # Border of goal set
        border_condition = _And( C > beta, dG )
        lie_condition = _And(_And(self.goal, _Not(B)), Cdot <=0)
        F = _And(border_condition, lie_condition)
        s = verifier.new_solver()
        res, timedout = verifier.solve_with_timeout(s, F)
        return verifier.is_sat(res)


# TODO: in devel
class CtrlLyapunov(Certificate):
    """
    Certifies stability and computes the control for CT models
    bool LLO: last layer of ones in network
    XD: Symbolic formula of domain

    """

    XD = "lie-&-pos"
    SD = XD

    def __init__(self, domains, **kw) -> None:
        self.llo = kw.get(CegisConfig.LLO.k, CegisConfig.LLO.v)
        self.domain = domains[Lyapunov.XD]
        self.bias = False

    # the Lyapunov certificate use a static Sdot, whereas we need to recompute Sdot at every iteration, since it
    # comes from another network
    def learn(self, learner: learner.Learner, optimizer: Optimizer, S: list, Sdot: list, f_torch: callable) -> dict:
        """
        :param learner: learner object
        :param optimizer: torch optimiser
        :param S: list of tensors of data
        :param Sdot: list of tensors containing f(data) -- actually NOT used, just for compatibility
        :return: --
        """

        samples = S[Lyapunov.SD]
        samples_dot = f_torch(samples)
        assert len(samples) == len(samples_dot)
        batch_size = len(samples)
        learn_loops = 1000
        margin = 0 * 0.01

        for t in range(learn_loops):
            optimizer.zero_grad()

            # recompute Sdot with new controller
            samples_dot = f_torch(samples)
            # print(samples_dot[0])
            
            V, Vdot, circle = learner.forward(samples, samples_dot)

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
        for cs in ({Lyapunov.SD: lyap_condition},):
            yield cs


class CtrlBarrier(Certificate):
    """
    Certifies Safety for CT and DT models

    Arguments:
    domains {dict}: dictionary of string:domains pairs for an initial set, unsafe set and domain

    Keyword Arguments:
    SYMMETRIC_BELT {bool}: sets belt symmetry

    """

    XD = "lie"
    XI = "init"
    XU = "unsafe"
    SD = XD
    SI = XI
    SU = XU

    def __init__(self, domains, **kw) -> None:
        self.domain = domains["lie"]
        self.initial_s = domains["init"]
        self.unsafe_s = domains["unsafe"]
        self.SYMMETRIC_BELT = kw.get(
            CegisConfig.SYMMETRIC_BELT.k, CegisConfig.SYMMETRIC_BELT.v
        )
        self.bias = True

    # the Barrier certificate use a static Sdot, whereas we need to recompute Sdot at every iteration, since it
    # comes from another network
    def learn(
        self, learner: learner.Learner, optimizer: Optimizer, S: dict, Sdot: dict, f_torch: callable
    ) -> dict:
        """
        :param learner: learner object
        :param optimizer: torch optimizer
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
        S_cat = torch.cat([s for s in S.values()])
        Sdot_ = f_torch(S_cat)
        assert len(S_cat) == len(Sdot_)


        for t in range(learn_loops):
            optimizer.zero_grad()

            # recompute Sdot with new controller
            samples_dot = f_torch(S_cat)

            # This seems slightly faster
            B, Bdot, _ = learner.forward(S_cat, samples_dot)
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

                loss = (
                    loss
                    + (relu6(dB_belt + 0 * margin)).mean()
                    - slope * relu6(-dB_belt).mean()
                )

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
            {Barrier.SI: inital_constr, Barrier.SU: unsafe_constr},
            {Barrier.SD: lie_constr},
        ):
            yield cs

class CtrlRWS(Certificate):
    XD = "lie"
    XI = "init"
    XS = "safe"
    dXS = "safe_border"
    XG = "goal"
    SD = XD
    SI = XI
    SS = XS
    SG = XG

    # Reach While stay must satisfy:
    # \forall x in XI, V <= 0,
    # \forall x in boundary of XS, V > 0,
    # \forall x in A \ XG, dV/dt <= 0
    # A = {x \in XS| V <=0 }

    def __init__(self, domains, **kw) -> None:
        # TODO: Make set labels constants of the class
        self.domain = domains[RWS.XD]
        self.initial_s = domains[RWS.XI]
        self.safe_s = domains[RWS.XS]
        self.safe_border = domains[RWS.dXS]
        self.goal = domains[RWS.XG]
        self.bias = True

    def learn(
        self, learner: learner.Learner, optimizer: Optimizer, S: list, Sdot: list, f_torch: callable
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
        i1 = S[RWS.XD].shape[0]
        i2 = S[RWS.XI].shape[0]
        S_cat = torch.cat(
            (S[RWS.XD], S[RWS.XI], S[RWS.XS])
        )
        Sdot_cat = f_torch(S_cat)

        for t in range(learn_loops):
            optimizer.zero_grad()

            Sdot_cat = f_torch(S_cat)

            B, Bdot, _ = learner.forward(S_cat, Sdot_cat)
            B_d, Bdot_d, = (
                B[:i1],
                Bdot[:i1],
            )
            B_i = B[i1 : i1 + i2]
            B_u = B[i1 + i2 :]

            learn_accuracy = sum(B_i <= -margin).item() + sum(B_u >= margin).item()
            percent_accuracy_init_unsafe = (
                learn_accuracy
                * 100
                / (len(S[RWS.XI]) + len(S[RWS.XS]))
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
        # C > 0 if x \in safe border
        safe_constr = _And(C <= 0, self.safe_border)

        # lie_constr = And(C >= -0.05, C <= 0.05, Cdot > 0)
        gamma = 0
        lie_constr = _And(_And(C >= 0, _Not(self.goal)), Cdot > gamma)

        # add domain constraints
        inital_constr = _And(initial_constr, self.domain)
        safe_constr = _And(safe_constr, self.domain)
        lie_constr = _And(lie_constr, self.domain)

        
        for cs in (
            {RWS.XI: inital_constr, RWS.XS: safe_constr},
            {RWS.XD: lie_constr},
        ):
            yield cs

class CtrlRSWS(Certificate):
    
    XD = "lie"
    XI = "init"
    XU = "unsafe_border"
    XS = "safe"
    XG = "goal"
    dXG = "goal_border"
    SD = XD
    SI = XI
    SU = "unsafe"
    SG = XG

    def __init__(self, domains, **kw) -> None:
        self.domain = domains[RSWS.XD]
        self.initial_s = domains[RSWS.XI]
        self.unsafe_border = domains[RSWS.XU]
        self.safe = domains[RSWS.XS]
        self.goal = domains[RSWS.XG]
        self.goal_border = domains[RSWS.dXG]
        self.bias = True

    def learn(self, learner: learner.Learner, optimizer: Optimizer, S: list, Sdot: list, f_torch: callable) -> dict:
        """learning function for RSWS

        Args:
            learner (learner.Learner): learner object
            optimizer (Optimizer): torch optimizer 
            S (list): list of tensors of the state
            Sdot (list): list of tensors of the state derivative

        Returns:
            dict: empty dict
        """    
        assert len(S) == len(Sdot)

        learn_loops = 1000
        margin = 0.1
        condition_old = False
        i1 = S[RSWS.XD].shape[0]
        i2 = S[RSWS.XI].shape[0]
        S_cat = torch.cat(
            (S[RSWS.XD], S[RSWS.XI], S[RSWS.SU])
        )
        Sdot_ = f_torch(S_cat)
        assert(len(S_cat) == len(Sdot_))
        for t in range(learn_loops):
            optimizer.zero_grad()

            Sdot_cat = f_torch(S_cat)

            B, Bdot, _ = learner.forward(S_cat, Sdot_cat)
            B_d, Bdot_d, = (
                B[:i1],
                Bdot[:i1],
            )
            B_i = B[i1 : i1 + i2]
            B_u = B[i1 + i2 :]

            learn_accuracy = sum(B_i <= -margin).item() + sum(B_u >= margin).item()
            percent_accuracy_init_unsafe = (
                learn_accuracy
                * 100
                / (len(S[RSWS.XI]) + len(S[RSWS.SU]))
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
                        "- accuracy init-safe:",
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
        """returns the constraints of the certificate

        Args:
            verifier (Verifier): verifier object
            C: SMT formula of certificate function
            Cdot: SMT formula of certificate lie derivative

        Yields:
            Generator: tuple of dictionaries of certificate conditons
        """        
        _And = verifier.solver_fncts()["And"]
        _Not = verifier.solver_fncts()["Not"]
        # Cdot <= 0 in C == 0
        # C <= 0 if x \in initial
        initial_constr = _And(C > 0, self.initial_s)
        # C > 0 if x \in safe border
        safe_constr = _And(C <= 0, self.unsafe_border)

        # lie_constr = And(C >= -0.05, C <= 0.05, Cdot > 0)
        gamma = 0
        lie_constr = _And(_And(C >= 0, _Not(self.goal)), Cdot > gamma)

        # add domain constraints
        inital_constr = _And(initial_constr, self.domain)
        safe_constr = _And(safe_constr, self.domain)
        lie_constr = _And(lie_constr, self.domain)

        for cs in (
            {RSWS.XI: inital_constr, RSWS.XU: safe_constr},
            {RSWS.XD: lie_constr},
        ):
            yield cs
    
    def stay_in_goal_check(self, verifier, C, Cdot):
        """Checks if the system stays in the goal region. True if it stays in the goal region, False otherwise.

        This check involves finding a beta such that:

        \forall x in border XG: V > \beta
        \forall x in XG \ int(B): dV/dt <= 0
        B = {x in XS | V <= \beta}

        Args:
            verifier (Verifier): verifier object
            C: SMT formula of certificate function
            Cdot: SMT formula of certificate lie derivative

        Returns:
            bool: True if sat
        """
        _And = verifier.solver_fncts()["And"]
        _Not = verifier.solver_fncts()["Not"]
        beta = verifier.new_vars(1, base='b')[0]
        B = _And(C <= beta, _And(self.domain, self.safe)) 
        dG = self.goal_border # Border of goal set
        border_condition = _And( C > beta, dG )
        lie_condition = _And(_And(self.goal, _Not(B)), Cdot <=0)
        F = _And(border_condition, lie_condition)
        s = verifier.new_solver()
        res, timedout = verifier.solve_with_timeout(s, F)
        return verifier.is_sat(res)




def get_certificate(certificate: CertificateType):
    if certificate == CertificateType.LYAPUNOV:
        return Lyapunov
    elif certificate == CertificateType.BARRIER:
        return Barrier
    elif certificate == CertificateType.BARRIER_LYAPUNOV:
        return BarrierLyapunov
    elif certificate == CertificateType.RWS:
        return RWS
    elif certificate == CertificateType.RSWS:
        return RSWS
    elif certificate == CertificateType.CTRLLYAP:
        return CtrlLyapunov
    elif certificate == CertificateType.CTRLBARR:
        return CtrlBarrier
    elif certificate == CertificateType.CTRLRWS:
        return CtrlRWS
    elif certificate == CertificateType.CTRLRSWS:
        return CtrlRSWS
    else:
        raise ValueError("Unknown certificate type {}".format(certificate))