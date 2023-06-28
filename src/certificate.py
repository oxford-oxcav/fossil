# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Generator, Type

import torch
from torch.optim import Optimizer

import src.control as control
import src.learner as learner
from src.consts import CegisConfig, CertificateType
from src.utils import vprint

XD = "lie"
XI = "init"
XU = "unsafe"
XS = "safe"
XG = "goal"
XG_BORDER = "goal_border"
XS_BORDER = "safe_border"
XF = "final"


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

    def __init__(self, domains, config: CegisConfig) -> None:
        self.domain = domains[XD]
        self.bias = False
        self.pos_def = False
        self.llo = CegisConfig.LLO
        self.control = config.CTRLAYER is not None

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
            learn_accuracy = (Vdot <= -margin).count_nonzero().item()
            loss = (leaky_relu(Vdot + margin * circle)).mean()
        else:
            learn_accuracy = 0.5 * (
                (Vdot <= -margin).count_nonzero().item()
                + (V >= margin).count_nonzero().item()
            )
            loss = (leaky_relu(Vdot + margin * circle)).mean() + (
                leaky_relu(-V + margin * circle)
            ).mean()

        return loss, learn_accuracy

    def learn(
        self,
        learner: learner.LearnerNN,
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

        batch_size = len(S[XD])
        learn_loops = 1000
        samples = S[XD]

        if f_torch:
            samples_dot = f_torch(samples)
        else:
            samples_dot = Sdot[XD]

        assert len(samples) == len(samples_dot)

        for t in range(learn_loops):
            optimizer.zero_grad()
            if self.control:
                samples_dot = f_torch(samples)

            V, Vdot, circle = learner.get_all(samples, samples_dot)

            loss, learn_accuracy = self.compute_loss(V, Vdot, circle)

            if self.control:
                loss = loss + control.nonzero_loss2(samples, samples_dot)
                # loss = loss + control.cosine_reg(samples, samples_dot)

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
        for cs in ({XD: lyap_condition},):
            yield cs


class Barrier(Certificate):
    """
    Certifies Safety for CT models

    Arguments:
    domains {dict}: dictionary of string:domains pairs for a initial set, unsafe set and domain

    Keyword Arguments:
    SYMMETRIC_BELT {bool}: sets belt symmetry

    """

    def __init__(self, domains, config: CegisConfig) -> None:
        self.domain = domains[XD]
        self.initial_s = domains[XI]
        self.unsafe_s = domains[XU]
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

        ### We spend A LOT of time computing the accuracies and belt percent.
        ### Changing from builtins sum to torch.sum() makes the whole code 4x faster.
        learn_accuracy = (B_i <= -margin).count_nonzero().item() + (
            B_u >= margin
        ).count_nonzero().item()
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
            learn_accuracy = (
                learn_accuracy + ((dB_belt <= -margin).count_nonzero()).item()
            )
            percent_belt = (
                100 * ((dB_belt <= -margin).count_nonzero()).item() / dB_belt.shape[0]
            )

            lie_loss = (relu6(dB_belt + 0 * margin)).mean() - slope * relu6(
                -dB_belt
            ).mean()
            loss = loss + lie_loss

        return loss, percent_accuracy_init_unsafe, percent_belt, len(belt_index)

    def learn(
        self,
        learner: learner.LearnerNN,
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
        i1 = S[XD].shape[0]
        i2 = S[XI].shape[0]
        # samples = torch.cat([s for s in S.values()])
        label_order = [XD, XI, XU]
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
            (
                B_d,
                Bdot_d,
            ) = (
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
            {XI: inital_constr, XU: unsafe_constr},
            {XD: lie_constr},
        ):
            yield cs


class BarrierAlt(Certificate):
    """
    Certifies Safety of a model  using Lie derivative everywhere.

    Works for continuous and discrete models.

    Arguments:
    domains {dict}: dictionary of string: domains pairs for a initial set, unsafe set and domain


    """

    def __init__(self, domains, config: CegisConfig) -> None:
        self.domain = domains[XD]
        self.initial_s = domains[XI]
        self.unsafe_s = domains[XU]
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

        learn_accuracy = (B_i <= -margin).count_nonzero().item() + (
            B_u >= margin
        ).count_nonzero().item()
        percent_accuracy_init_unsafe = learn_accuracy * 100 / (len(B_u) + len(B_i))
        slope = 1 / 10**4  # (learner.orderOfMagnitude(max(abs(Vdot)).detach()))
        relu6 = torch.nn.ReLU6()
        p = 1
        init_loss = (torch.relu(B_i + margin) - slope * relu6(-B_i + margin)).mean()
        unsafe_loss = (torch.relu(-B_u + margin) - slope * relu6(B_u + margin)).mean()

        lie_loss = (relu6(Bdot_d + margin)).mean()

        # set two belts
        percent_belt = 0

        lie_accuracy = (
            100 * ((Bdot_d <= -margin).count_nonzero()).item() / Bdot_d.shape[0]
        )

        loss = init_loss + unsafe_loss + lie_loss
        return percent_accuracy_init_unsafe, percent_belt, lie_accuracy, loss

    def learn(
        self,
        learner: learner.LearnerNN,
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
        i1 = S[XD].shape[0]
        i2 = S[XI].shape[0]
        # samples = torch.cat([s for s in S.values()])
        label_order = [XD, XI, XU]
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
            (
                B_d,
                Bdot_d,
            ) = (
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
            {XI: inital_constr, XU: unsafe_constr},
            {XD: lie_constr},
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

    def __init__(self, domains, config: CegisConfig) -> None:
        """initialise the RWS certificate
        Domains should contain:
            XI: compact initial set
            XS: compact safe set
            dXS: safe border
            XG: compact goal set
            XD: whole domain

        Data sets for learn should contain:
            SI: points from XI
            SU: points from XD \ XS
            SD: points from XS \ XG (domain less unsafe and goal set)

        """
        self.domain = domains[XD]
        self.initial = domains[XI]
        self.safe = domains[XS]
        self.safe_border = domains[XS_BORDER]
        self.goal = domains[XG]
        self.bias = True

    def compute_loss(self, V_i, V_u, V_d, Vdot_d):
        margin = 0
        learn_accuracy = (V_i <= -margin).count_nonzero().item() + (
            V_u >= margin
        ).count_nonzero().item()
        percent_accuracy_init_unsafe = learn_accuracy * 100 / (len(V_i) + len(V_u))
        slope = 0  # 1 / 10**4  # (learner.orderOfMagnitude(max(abs(Vdot)).detach()))
        relu = torch.nn.Softplus()

        init_loss = (torch.relu(V_i + margin) - slope * relu(-V_i + margin)).mean()
        unsafe_loss = (torch.relu(-V_u + margin) - slope * relu(V_u + margin)).mean()
        loss = init_loss + unsafe_loss

        # Penalise pos lie derivative for all points not in unsafe set or goal set where V <= 0
        # This assumes V_d has no points in the unsafe set or goal set - is this reasonable?

        lie_index = torch.nonzero(V_d < -margin)
        if lie_index.nelement() != 0:
            A_lie = torch.index_select(Vdot_d, dim=0, index=lie_index[:, 0])
            lie_accuracy = (
                ((A_lie <= -margin).count_nonzero()).item() * 100 / A_lie.shape[0]
            )

            lie_loss = (relu(A_lie + 0 * margin)).mean() - slope * relu(-Vdot_d).mean()
            loss = loss + lie_loss
        else:
            lie_accuracy = 0.0

        loss = loss
        return percent_accuracy_init_unsafe, loss, lie_accuracy

    def learn(
        self,
        learner: learner.LearnerNN,
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
        i1 = S[XD].shape[0]
        i2 = S[XI].shape[0]
        label_order = [XD, XI, XU]
        samples = torch.cat([S[label] for label in label_order])
        # samples = torch.cat((S[XD], S[XI], S[XU]))

        if f_torch:
            samples_dot = f_torch(samples)
        else:
            samples_dot = torch.cat([Sdot[label] for label in label_order])

        for t in range(learn_loops):
            optimizer.zero_grad()
            if f_torch:
                samples_dot = f_torch(samples)

            V, Vdot, _ = learner.get_all(samples, samples_dot)
            (
                V_d,
                Vdot_d,
            ) = (
                V[:i1],
                Vdot[:i1],
            )
            V_i = V[i1 : i1 + i2]
            V_u = V[i1 + i2 :]

            percent_accuracy_init_unsafe, loss, lie_accuracy = self.compute_loss(
                V_i, V_u, V_d, Vdot_d
            )

            if f_torch:
                s = samples[:i1]
                sdot = samples_dot[:i1]
                loss = loss + 0.05 * control.ridge_reg(s, sdot)
                # loss = loss + control.nonzero_loss2(samples, samples_dot)
                # loss = loss + control.cosine_reg(samples, samples_dot)
                # W = optimizer.param_groups[1]["params"]
                # loss = loss + 0.05 * control.ridge_reg2(W)
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
        initial_constr = _And(C > 0, self.initial)
        # C > 0 if x \in safe border
        unsafe_constr = _And(C <= 0, self.safe_border)

        # lie_constr = And(C >= -0.05, C <= 0.05, Cdot > 0)
        gamma = 0

        # Define A as the set of points where C <= 0, within the domain, not in the goal set, and not in the unsafe set
        A = _And(C <= 0, self.safe, _Not(self.goal))
        lie_constr = _And(A, Cdot > gamma)

        # add domain constraints
        inital_constr = _And(initial_constr, self.domain)
        unsafe_constr = _And(unsafe_constr, self.domain)

        for cs in (
            {XI: inital_constr, XU: unsafe_constr},
            {XD: lie_constr},
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

    def __init__(self, domains, config: CegisConfig) -> None:
        """initialise the RSWS certificate
        Domains should contain:
            XI: compact initial set
            XS: compact safe set
            dXS: safe border
            XG: compact goal set
            XD: whole domain

        Data sets for learn should contain:
            SI: points from XI
            SU: points from XD \ XS
            SD: points from XS \ XG (domain less unsafe and goal set)

        """
        self.domain = domains[XD]
        self.initial = domains[XI]
        self.safe = domains[XS]
        self.safe_border = domains[XS_BORDER]
        self.goal = domains[XG]
        self.goal_border = domains[XG_BORDER]
        self.bias = True

    def compute_beta_loss(self, beta, V_d, Vdot_d):
        """Compute the loss for the beta condition"""
        lie_index = torch.nonzero(V_d <= beta)
        relu = torch.nn.Softplus()
        margin = 0
        slope = 0
        if lie_index.nelement() != 0:
            beta_lie = torch.index_select(Vdot_d, dim=0, index=lie_index[:, 0])
            beta_lie_loss = (relu(beta_lie + 0 * margin)).mean() - slope * relu(
                -Vdot_d
            ).mean()
            accuracy = (beta_lie <= 0).count_nonzero().item() * 100 / beta_lie.shape[0]
        else:
            beta_lie_loss = 0
        return beta_lie_loss

    def learn(
        self,
        learner: learner.LearnerNN,
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
        lie_indices = 0, S[XD].shape[0]
        init_indices = lie_indices[1], lie_indices[1] + S[XI].shape[0]
        unsafe_indices = init_indices[1], init_indices[1] + S[XU].shape[0]
        goal_border_indices = (
            unsafe_indices[1],
            unsafe_indices[1] + S[XG_BORDER].shape[0],
        )
        goal_indices = (
            goal_border_indices[1],
            goal_border_indices[1] + S[XG].shape[0],
        )
        # Setting label order allows datasets to be passed in any order
        label_order = [XD, XI, XU, XG_BORDER, XG]
        samples = torch.cat([S[label] for label in label_order])

        if f_torch:
            samples_dot = f_torch(samples)
        else:
            samples_dot = torch.cat([Sdot[label] for label in label_order])

        for t in range(learn_loops):
            optimizer.zero_grad()
            if f_torch:
                samples_dot = f_torch(samples)

            V, Vdot, _ = learner.get_all(samples, samples_dot)
            (
                V_d,
                Vdot_d,
            ) = (
                V[: lie_indices[1]],
                Vdot[: lie_indices[1]],
            )
            V_i = V[init_indices[0] : init_indices[1]]
            V_u = V[unsafe_indices[0] : unsafe_indices[1]]
            S_dg = samples[goal_border_indices[0] : goal_border_indices[1]]
            V_g = V[goal_indices[0] : goal_indices[1]]
            Vdot_g = Vdot[goal_indices[0] : goal_indices[1]]

            percent_accuracy_init_unsafe, loss, lie_accuracy = self.compute_loss(
                V_i, V_u, V_d, Vdot_d
            )

            beta = learner.compute_minimum(S_dg)[0]
            beta_loss = self.compute_beta_loss(beta, V_g, Vdot_g)
            loss = loss + beta_loss

            if f_torch:
                s = samples[: lie_indices[1]]
                sdot = samples_dot[: lie_indices[1]]
                loss = loss + 0.0075 * (s @ sdot.T).diag().mean()

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

    def stay_in_goal_check(self, verifier, C, Cdot, beta=-10):
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
        dG = self.goal_border  # Border of goal set
        # Solving these separately means we don't have to deal with
        # spurious counterexamples
        border_condition = _And(C <= beta, dG)
        s_border = verifier.new_solver()
        res1, timedout = verifier.solve_with_timeout(s_border, border_condition)

        B = _And(self.safe, C < beta)
        XG_less_B = _And(self.goal, _Not(B))
        lie_condition = _And(XG_less_B, Cdot > 0)
        s_lie = verifier.new_solver()
        res2, timedout = verifier.solve_with_timeout(s_lie, lie_condition)

        if verifier.is_unsat(res1) and verifier.is_unsat(res2):
            return True, None
        elif verifier.is_sat(res1) and verifier.is_unsat(res2):
            # Border condition is sat, beta too high
            return False, "decrease"
        elif verifier.is_unsat(res1) and verifier.is_sat(res2):
            # Lie condition is sat, beta too low
            return False, "increase"
        else:
            assert verifier.is_sat(res1) and verifier.is_sat(res2)
            return False, "both"
            # raise RuntimeError("Both conditions are sat")

    def beta_search(self, learner, verifier, C, Cdot, S):
        """Searches for a beta to prove that the system stays in the goal region.

        Args:
            learner (Learner): learner object
            verifier (Verifier): verifier object
            C: SMT formula of certificate function
            Cdot: SMT formula of certificate lie derivative
            S (dict): data sets

        Returns:
            bool, float: res, beta
        """
        # In Verdier, Mazo they do a line search over Beta. I'm not sure how to do that currently.
        # I'm not even sure if beta should be negative or positive, or on its scale.
        # We could also do a exists forall query with dreal, but it would scale poorly.

        # beta_upper = learner.compute_minimum(S["goal"])[0]
        # beta_lower = learner.compute_minimum(S["safe"])[0]
        beta_upper = 1000
        beta_lower = -1000000
        beta_guess = (beta_upper + beta_lower) / 2
        c = 0
        while c < 50:
            res, bound = self.stay_in_goal_check(verifier, C, Cdot, beta_guess)
            if res:
                return True
            else:
                if bound == "decrease":
                    beta_upper = beta_guess
                    beta_guess = (beta_upper + beta_lower) / 2
                elif bound == "increase":
                    beta_lower = beta_guess
                    beta_guess = (beta_upper + beta_lower) / 2
                elif bound == "both":
                    l = torch.rand(1).item()
                    beta_guess = l * beta_upper + (1 - l) * beta_lower
                    c += 1  # We should only do this so many times, and then give up
                print(beta_guess)
        return False


class StableSafe(Certificate):
    """Certificate to prove stable while safe"""

    def __init__(self, domains, config: CegisConfig) -> None:
        self.domain = domains[XD]
        self.initial_s = domains[XI]
        self.unsafe_s = domains[XU]
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
            learn_accuracy = (Vdot <= -margin).count_nonzero().item()
            loss = (leaky_relu(Vdot + margin * circle)).mean()
        else:
            learn_accuracy = 0.5 * (
                (Vdot <= -margin).count_nonzero().item()
                + (V >= margin).count_nonzero().item()
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
        learn_accuracy = (B_i <= -margin).count_nonzero().item() + (
            B_u >= margin
        ).count_nonzero().item()
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
            learn_accuracy = (
                learn_accuracy + ((dB_belt <= -margin).count_nonzero()).item()
            )
            percent_belt = (
                100 * ((dB_belt <= -margin).count_nonzero()).item() / dB_belt.shape[0]
            )

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
        i1 = S[XD].shape[0]
        i2 = S[XI].shape[0]
        label_order = [XD, XI, XU]
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
            (
                B_d,
                Bdot_d,
            ) = (
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

        return {XD: lyap_condition}

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
            {XI: inital_constr, XU: unsafe_constr},
            {XD: lie_constr},
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


class DoubleCertificate(Certificate):
    """In Devel class for synthesising any two certificates together"""

    def __init__(self, domains, config: CegisConfig):
        self.certificate1 = get_certificate(config.CERTIFICATE)(domains, config)
        self.certificate2 = get_certificate(config.CERTIFICATE_ALT)(domains, config)

    def compute_loss(self, C1, C2, Cdot1, Cdot2):
        pass

    def learn(
        self, learner: tuple, optimizer: Optimizer, S: dict, Sdot: dict, f_torch=None
    ):
        pass

    def get_constraints(self, verifier, C, Cdot) -> Generator:
        """
        :param verifier: verifier object
        :param C: tuple containing SMT formula of Lyapunov function and barrier function
        :param Cdot: tuple containing SMT formula of Lyapunov lie derivative and barrier lie derivative

        """
        C1, C2 = C
        Cdot1, Cdot2 = Cdot
        cert1_cs = self.certificate1.get_constraints(verifier, C1, Cdot1)
        cert2_cs = self.certificate2.get_constraints(verifier, C2, Cdot2)
        for cs in (*cert1_cs, *cert2_cs):
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
