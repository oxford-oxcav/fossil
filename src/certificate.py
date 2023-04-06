# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from torch.optim import Optimizer
from typing import Generator, Type

import torch
from torch.optim import Optimizer

import src.learner as learner
from src.shared.utils import vprint
from src.shared.consts import CertificateType, CegisConfig


class Certificate:
    def __init__(self) -> None:
        pass

    def learn(self, optimizer: Optimizer, S: list, Sdot: list, f_torch=None) -> dict:
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

    def __init__(self, domains, config: CegisConfig) -> None:
        self.domain = domains[Lyapunov.XD]
        self.bias = False
        self.pos_def = False
        self.llo = CegisConfig.LLO

    def compute_loss(
        self, V: torch.Tensor, Vdot: torch.Tensor, circle: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        """_summary_

        Args:
            V (torch.Tensor): Lyapunov samples over domain
            Vdot (torch.Tensor): Lyapunov derivative samples over domain
            circle (torch.Tensor): Circle

        Returns:
            tuple[torch.Tensor, float]: loss and accuracy
        """
        margin = 0 * 0.01

        slope = 10 ** (learner.LearnerNN.order_of_magnitude(max(abs(Vdot)).detach()))
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

        return loss, learn_accuracy

    def learn(
        self,
        learner: learner.Learner,
        optimizer: Optimizer,
        S: list,
        Sdot: list,
        f_torch=None,
    ) -> dict:
        """
        :param learner: learner object
        :param optimizer: torch optimiser
        :param S: list of tensors of data
        :param Sdot: list of tensors containing f(data)
        :return: --
        """

        batch_size = len(S[Lyapunov.SD])
        learn_loops = 1000
        samples = S[Lyapunov.SD]

        if f_torch:
            samples_dot = f_torch(samples)
        else:
            samples_dot = Sdot[Lyapunov.SD]

        assert len(samples) == len(samples_dot)

        for t in range(learn_loops):
            optimizer.zero_grad()
            if f_torch:
                samples_dot = f_torch(samples)

            V, Vdot, circle = learner.get_all(samples, samples_dot)

            loss, learn_accuracy = self.compute_loss(V, Vdot, circle)

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

        if self.llo:
            # V is positive definite by construction
            lyap_negated = Vdot > 0
        else:
            lyap_negated = _Or(V <= 0, Vdot > 0)
        lyap_condition = _And(self.domain, lyap_negated)
        for cs in ({Lyapunov.SD: lyap_condition},):
            yield cs


class Barrier(Certificate):
    """
    Certifies Safety for CT models

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

    def __init__(self, domains, config: CegisConfig) -> None:
        self.domain = domains[Barrier.XD]
        self.initial_s = domains[Barrier.XI]
        self.unsafe_s = domains[Barrier.XU]
        self.SYMMETRIC_BELT = config.SYMMETRIC_BELT
        self.bias = True

    def compute_loss(
        self,
        B_i: torch.Tensor,
        B_u: torch.Tensor,
        B_d: torch.Tensor,
        Bdot_d: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Computes loss function for Barrier certificate.

        Also computes accuracy of the current model.

        Args:
            B_i (torch.Tensor): Barrier values for initial set
            B_u (torch.Tensor): Barrier values for unsafe set
            B_d (torch.Tensor): Barrier values for domain
            Bdot_d (torch.Tensor): Barrier derivative values for domain

        Returns:
            tuple[torch.Tensor, float]: loss and accuracy
        """
        margin = 0
        slope = 1 / 10**4
        learn_accuracy = sum(B_i <= -margin).item() + sum(B_u >= margin).item()
        percent_accuracy_init_unsafe = learn_accuracy * 100 / (len(B_u) + len(B_i))

        relu6 = torch.nn.Softplus()
        init_loss = (torch.relu(B_i + margin) - slope * relu6(-B_i + margin)).mean()
        unsafe_loss = (torch.relu(-B_u + margin) - slope * relu6(B_u + margin)).mean()
        loss = init_loss + unsafe_loss

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

            lie_loss = (relu6(dB_belt + 0 * margin)).mean() - slope * relu6(
                -dB_belt
            ).mean()
            loss = loss + lie_loss

        return loss, percent_accuracy_init_unsafe, percent_belt, len(belt_index)

    def learn(
        self,
        learner: learner.Learner,
        optimizer: Optimizer,
        S: dict,
        Sdot: dict,
        f_torch=None,
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
        condition_old = False
        i1 = S[Barrier.XD].shape[0]
        i2 = S[Barrier.XI].shape[0]
        # samples = torch.cat([s for s in S.values()])
        label_order = [self.XD, self.XI, self.XU]
        samples = torch.cat([S[label] for label in label_order])

        if f_torch:
            samples_dot = f_torch(samples)
        else:
            # samples_dot = torch.cat([s for s in Sdot.values()])
            samples_dot = torch.cat([Sdot[label] for label in label_order])

        for t in range(learn_loops):
            optimizer.zero_grad()

            if f_torch:
                samples_dot = f_torch(samples)

            # This seems slightly faster
            B, Bdot, _ = learner.get_all(samples, samples_dot)
            B_d, Bdot_d, = (
                B[:i1],
                Bdot[:i1],
            )
            B_i = B[i1 : i1 + i2]
            B_u = B[i1 + i2 :]

            (
                loss,
                accuracy_init_unsafe,
                accuracy_belt,
                N_belt,
            ) = self.compute_loss(B_i, B_u, B_d, Bdot_d)

            if t % int(learn_loops / 10) == 0 or learn_loops - t < 10:
                vprint(
                    (
                        t,
                        "- loss:",
                        loss.item(),
                        "- accuracy init-unsafe:",
                        accuracy_init_unsafe,
                        "- accuracy belt:",
                        accuracy_belt,
                        "- points in belt:",
                        N_belt,
                    ),
                    learner.verbose,
                )

            if accuracy_init_unsafe == 100 and accuracy_belt >= 99.9:
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


class BarrierAlt(Certificate):
    """
    Certifies Safety of a model  using Lie derivative everywhere.

    Works for continuous and discrete models.

    Arguments:
    domains {dict}: dictionary of string: domains pairs for a initial set, unsafe set and domain


    """

    XD = "lie"
    XI = "init"
    XU = "unsafe"
    SD = XD
    SI = XI
    SU = XU

    def __init__(self, domains, config: CegisConfig) -> None:
        self.domain = domains[BarrierAlt.XD]
        self.initial_s = domains[BarrierAlt.XI]
        self.unsafe_s = domains[BarrierAlt.XU]
        self.bias = True

    def compute_loss(
        self,
        B_i: torch.Tensor,
        B_u: torch.Tensor,
        B_d: torch.Tensor,
        Bdot_d: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Computes loss function for Barrier certificate.

        Also computes accuracy of the current model.

        Args:
            B_i (torch.Tensor): Barrier values for initial set
            B_u (torch.Tensor): Barrier values for unsafe set
            B_d (torch.Tensor): Barrier values for domain
            Bdot_d (torch.Tensor): Barrier derivative values for domain

        Returns:
            tuple[torch.Tensor, float]: loss and accuracy
        """
        margin = 0.1

        learn_accuracy = sum(B_i <= -margin).item() + sum(B_u >= margin).item()
        percent_accuracy_init_unsafe = learn_accuracy * 100 / (len(B_u) + len(B_i))
        slope = 1 / 10**4  # (learner.orderOfMagnitude(max(abs(Vdot)).detach()))
        relu6 = torch.nn.ReLU6()
        p = 1
        init_loss = (torch.relu(B_i + margin) - slope * relu6(-B_i + margin)).mean()
        unsafe_loss = (torch.relu(-B_u + margin) - slope * relu6(B_u + margin)).mean()

        lie_loss = (relu6(Bdot_d + margin)).mean()

        # set two belts
        percent_belt = 0

        lie_accuracy = 100 * (sum(Bdot_d <= -margin)).item() / Bdot_d.shape[0]

        loss = init_loss + unsafe_loss + lie_loss
        return percent_accuracy_init_unsafe, percent_belt, lie_accuracy, loss

    def learn(
        self,
        learner: learner.Learner,
        optimizer: Optimizer,
        S: list,
        Sdot: list,
        f_torch=None,
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
        condition_old = False
        i1 = S[BarrierAlt.XD].shape[0]
        i2 = S[BarrierAlt.XI].shape[0]
        # samples = torch.cat([s for s in S.values()])
        label_order = [self.XD, self.XI, self.XU]
        samples = torch.cat([S[label] for label in label_order])

        if f_torch:
            samples_dot = f_torch(samples)
        else:
            # samples_dot = torch.cat([s for s in Sdot.values()])
            samples_dot = torch.cat([Sdot[label] for label in label_order])

        for t in range(learn_loops):
            optimizer.zero_grad()

            if f_torch:
                samples_dot = f_torch(samples)

            # permutation_index = torch.randperm(S[0].size()[0])
            # permuted_S, permuted_Sdot = S[0][permutation_index], S_dot[0][permutation_index]
            B, Bdot, _ = learner.get_all(samples, samples_dot)
            B_d, Bdot_d, = (
                B[:i1],
                Bdot[:i1],
            )
            B_i = B[i1 : i1 + i2]
            B_u = B[i1 + i2 :]

            (
                percent_accuracy_init_unsafe,
                percent_belt,
                lie_accuracy,
                loss,
            ) = self.compute_loss(B_i, B_u, B_d, Bdot_d)

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
            {BarrierAlt.SD: inital_constr, BarrierAlt.SU: unsafe_constr},
            {BarrierAlt.SD: lie_constr},
        ):
            yield cs


class RWS(Certificate):
    """Certificate to satisfy a reach-while-stay property.

    Reach While stay must satisfy:
    \forall x in XI, V <= 0,
    \forall x in boundary of XS, V > 0,
    \forall x in A \ XG, dV/dt <= 0
    A = {x \in XS| V <=0 }

    """

    XD = "lie"
    XI = "init"
    XS = "safe"
    dXS = "safe_border"
    XG = "goal"
    SD = XD
    SI = XI
    SS = XS
    SG = XG

    def __init__(self, domains, **kw) -> None:
        self.domain = domains[RWS.XD]
        self.initial_s = domains[RWS.XI]
        self.safe_s = domains[RWS.XS]
        self.safe_border = domains[RWS.dXS]
        self.goal = domains[RWS.XG]
        self.bias = True

    def compute_loss(self, Bdot_d, B_i, B_u):
        margin = 0.1
        learn_accuracy = sum(B_i <= -margin).item() + sum(B_u >= margin).item()
        percent_accuracy_init_unsafe = learn_accuracy * 100 / (len(B_i) + len(B_u))
        slope = 1 / 10**4  # (learner.orderOfMagnitude(max(abs(Vdot)).detach()))
        relu6 = torch.nn.ReLU6()
        # saturated_leaky_relu = torch.nn.ReLU6() - 0.01*torch.relu()
        loss = (torch.relu(B_i + margin) - slope * relu6(-B_i + margin)).mean() + (
            torch.relu(-B_u + margin) - slope * relu6(B_u + margin)
        ).mean()

        lie_accuracy = 100 * (sum(Bdot_d <= -margin)).item() / Bdot_d.shape[0]

        loss = loss - (relu6(-Bdot_d + margin)).mean()
        return percent_accuracy_init_unsafe, loss, lie_accuracy

    def learn(
        self,
        learner: learner.Learner,
        optimizer: Optimizer,
        S: list,
        Sdot: list,
        f_torch=None,
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
        condition_old = False
        i1 = S[RWS.XD].shape[0]
        i2 = S[RWS.XI].shape[0]
        # I think dicts remember insertion order now, though perhaps this should be done more thoroughly
        # TODO: FIXME
        # This is a really bad thing to do as it means the sets must be passed in this order within the dictionaries,
        # which is not a good thing to rely on. Must be fixed for all cetificates.
        label_order = [RWS.XD, RWS.XI, RWS.XS]
        samples = torch.cat([S[label] for label in label_order])
        # samples = torch.cat((S[RWS.XD], S[RWS.XI], S[RWS.XS]))

        if f_torch:
            samples_dot = f_torch(samples)
        else:
            samples_dot = torch.cat([Sdot[label] for label in label_order])

        for t in range(learn_loops):
            optimizer.zero_grad()
            if f_torch:
                samples_dot = f_torch(samples)

            B, Bdot, _ = learner.get_all(samples, samples_dot)
            B_d, Bdot_d, = (
                B[:i1],
                Bdot[:i1],
            )
            B_i = B[i1 : i1 + i2]
            B_u = B[i1 + i2 :]

            percent_accuracy_init_unsafe, loss, lie_accuracy = self.compute_loss(
                Bdot_d, B_i, B_u
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


class RSWS(RWS):
    """Reach and Stay While Stay Certificate

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

    def stay_in_goal_check(self, verifier, C, Cdot):
        """Checks if the system stays in the goal region.
        True if it stays in the goal region, False otherwise.

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
        beta = verifier.new_vars(1, base="b")[0]
        B = _And(C <= beta, _And(self.domain, self.safe))
        dG = self.goal_border  # Border of goal set
        border_condition = _And(C > beta, dG)
        lie_condition = _And(_And(self.goal, _Not(B)), Cdot <= 0)
        F = _And(border_condition, lie_condition)
        s = verifier.new_solver()
        res, timedout = verifier.solve_with_timeout(s, F)
        return verifier.is_sat(res)


class StableSafe(Certificate):
    """Certificate to prove stable while safe"""

    XD = SD = "lie"
    XI = SI = "init"
    XU = SU = "unsafe"

    def __init__(self, domains, config: CegisConfig) -> None:
        self.domain = domains["lie"]
        self.initial_s = domains["init"]
        self.unsafe_s = domains["unsafe"]
        self.SYMMETRIC_BELT = config.SYMMETRIC_BELT
        self.llo = config.LLO

    def compute_lyap_loss(
        self, V: torch.Tensor, Vdot: torch.Tensor, circle: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        """_summary_

        Args:
            V (torch.Tensor): Lyapunov samples over domain
            Vdot (torch.Tensor): Lyapunov derivative samples over domain
            circle (torch.Tensor): Circle

        Returns:
            tuple[torch.Tensor, float]: loss and accuracy
        """
        margin = 0 * 0.01

        slope = 10 ** (learner.LearnerNN.order_of_magnitude(max(abs(Vdot)).detach()))
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

        return loss, learn_accuracy

    def compute_barrier_loss(
        self,
        B_i: torch.Tensor,
        B_u: torch.Tensor,
        B_d: torch.Tensor,
        Bdot_d: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Computes loss function for Barrier certificate.

        Also computes accuracy of the current model.

        Args:
            B_i (torch.Tensor): Barrier values for initial set
            B_u (torch.Tensor): Barrier values for unsafe set
            B_d (torch.Tensor): Barrier values for domain
            Bdot_d (torch.Tensor): Barrier derivative values for domain

        Returns:
            tuple[torch.Tensor, float]: loss and accuracy
        """
        margin = 0
        slope = 1 / 10**4
        learn_accuracy = sum(B_i <= -margin).item() + sum(B_u >= margin).item()
        percent_accuracy_init_unsafe = learn_accuracy * 100 / (len(B_u) + len(B_i))

        relu6 = torch.nn.Softplus()
        init_loss = (torch.relu(B_i + margin) - slope * relu6(-B_i + margin)).mean()
        unsafe_loss = (torch.relu(-B_u + margin) - slope * relu6(B_u + margin)).mean()
        loss = init_loss + unsafe_loss

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

            lie_loss = (relu6(dB_belt + 0 * margin)).mean() - slope * relu6(
                -dB_belt
            ).mean()
            loss = loss + lie_loss

        return loss, percent_accuracy_init_unsafe, percent_belt, len(belt_index)

    def learn(
        self,
        learner: tuple,
        optimizer: Optimizer,
        S: dict,
        Sdot: dict,
        f_torch=None,
    ) -> dict:
        """
        :param learner: learner object
        :param optimizer: torch optimiser
        :param S: list of tensors of data
        :param Sdot: list of tensors containing f(data)
        :return: --
        """
        assert len(S) == len(Sdot)
        lyap_learner = learner[0]
        barrier_learner = learner[1]

        learn_loops = 1000
        condition_old = False
        i1 = S["lie"].shape[0]
        i2 = S["init"].shape[0]
        label_order = [self.XD, self.XI, self.XU]
        samples = torch.cat([S[label] for label in label_order])
        samples = torch.cat([s for s in S.values()])

        if f_torch:
            samples_dot = f_torch(samples)
        else:
            samples_dot = torch.cat([s for s in Sdot.values()])

        for t in range(learn_loops):
            optimizer.zero_grad()

            if f_torch:
                samples_dot = f_torch(samples)

            # This seems slightly faster
            V, Vdot, circle = lyap_learner.get_all(samples, samples_dot)
            B, Bdot, _ = barrier_learner.get_all(samples, samples_dot)
            B_d, Bdot_d, = (
                B[:i1],
                Bdot[:i1],
            )
            B_i = B[i1 : i1 + i2]
            B_u = B[i1 + i2 :]

            lyap_loss, lyap_acc = self.compute_lyap_loss(V, Vdot, circle)
            (
                b_loss,
                accuracy_init_unsafe,
                accuracy_belt,
                N_belt,
            ) = self.compute_barrier_loss(B_i, B_u, B_d, Bdot_d)

            loss = lyap_loss + b_loss

            if t % int(learn_loops / 10) == 0 or learn_loops - t < 10:
                vprint(
                    (
                        t,
                        "- loss:",
                        loss.item(),
                        "- lyap acc:",
                        lyap_acc,
                        "- accuracy init-unsafe:",
                        accuracy_init_unsafe,
                        "- accuracy belt:",
                        accuracy_belt,
                        "- points in belt:",
                        N_belt,
                    ),
                    lyap_learner.verbose,
                )

            if accuracy_init_unsafe == 100 and accuracy_belt >= 99.9:
                condition = True
            else:
                condition = False

            if condition and condition_old:
                break
            condition_old = condition

            loss.backward()
            optimizer.step()

        return {}

    def _get_lyap_constraints(self, verifier, V, Vdot):
        """Generates Lyapunov constraints

        Args:
            verifier (Verifier): Verifier object
            V: SMT formula of Lyapunov function
            Vdot: SMT formula of Lyapunov lie derivative

        Returns:
            constr (dict): Lyapunov constraints
        """
        _Or = verifier.solver_fncts()["Or"]
        _And = verifier.solver_fncts()["And"]

        if self.llo:
            # V is positive definite by construction
            lyap_negated = Vdot > 0
        else:
            lyap_negated = _Or(V <= 0, Vdot > 0)
        lyap_condition = _And(self.domain, lyap_negated)

        return {StableSafe.SD: lyap_condition}

    def _get_barrier_constraints(self, verifier, B, Bdot):
        """Generates Barrier constraints

        Args:
            verifier (Verifier): verifier object
            B: SMT formula of Barrier function
            Bdot: SMT formula of Barrier lie derivative

        Returns:
            constr: tuple of dictionaries of Barrier conditons
        """
        _And = verifier.solver_fncts()["And"]

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

        return (
            {StableSafe.SI: inital_constr, StableSafe.SU: unsafe_constr},
            {StableSafe.SD: lie_constr},
        )

    def get_constraints(self, verifier, C, Cdot) -> Generator:
        """
        :param verifier: verifier object
        :param C: tuple containing SMT formula of Lyapunov function and barrier function
        :param Cdot: tuple containing SMT formula of Lyapunov lie derivative and barrier lie derivative

        """
        V, B = C
        Vdot, Bdot = Cdot
        lyap_cs = self._get_lyap_constraints(verifier, V, Vdot)
        barrier_cs = self._get_barrier_constraints(verifier, B, Bdot)

        for cs in (lyap_cs, *barrier_cs):
            yield cs


def get_certificate(certificate: CertificateType) -> Type[Certificate]:
    if certificate == CertificateType.LYAPUNOV:
        return Lyapunov
    elif certificate == CertificateType.BARRIER:
        return Barrier
    elif certificate == CertificateType.BARRIERALT:
        return BarrierAlt
    elif certificate == CertificateType.RWS:
        return RWS
    elif certificate == CertificateType.RSWS:
        return RSWS
    elif certificate == CertificateType.STABLESAFE:
        return StableSafe
    else:
        raise ValueError("Unknown certificate type {}".format(certificate))
