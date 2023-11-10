# Copyright (c) 2023, Alessandro Abate, Alec Edwards,  Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable


import torch

import fossil as fs
from fossil import certificate


class CustomLyapunov(certificate.Certificate):
    """
    Certificies stability for CT and DT models
    bool LLO: last layer of ones in network
    XD: Symbolic formula of domain

    """

    def __init__(self, domains, config: fs.CegisConfig) -> None:
        self.domain = domains[fs.XD]
        self.bias = False

    def compute_loss(
        self, V: torch.Tensor, Vdot: torch.Tensor, circle: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """_summary_

        Args:
            V (torch.Tensor): Lyapunov samples over domain
            Vdot (torch.Tensor): Lyapunov derivative samples over domain
            circle (torch.Tensor): Circle

        Returns:
            tuple[torch.Tensor, float]: loss and accuracy
        """
        margin = 0
        slope = 10**2
        relu = torch.nn.LeakyReLU(1 / slope)
        # relu = torch.nn.Softplus()
        # compute loss function. if last layer of ones (llo), can drop parts with V
        learn_accuracy = 0.5 * (
            (Vdot <= -margin).count_nonzero().item()
            + (V >= margin).count_nonzero().item()
        )
        loss = (relu(Vdot + margin * circle)).mean() + (
            relu(-V + margin * circle)
        ).mean()
        accuracy = {"acc": learn_accuracy * 100 / Vdot.shape[0]}

        return loss, accuracy

    def learn(
        self,
        learner,
        optimizer,
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

        batch_size = len(S[fs.XD])
        learn_loops = 1000
        samples = S[fs.XD]

        if f_torch:
            samples_dot = f_torch(samples)
        else:
            samples_dot = Sdot[fs.XD]

        assert len(samples) == len(samples_dot)

        for t in range(learn_loops):
            optimizer.zero_grad()

            V, Vdot, circle = learner.get_all(samples, samples_dot)

            loss, learn_accuracy = self.compute_loss(V, Vdot, circle)

            # t>=1 ensures we always have at least 1 optimisation step
            if learn_accuracy["acc"] == 100 and t >= 1:
                break

            loss.backward()
            optimizer.step()

            if learner._take_abs:
                learner.make_final_layer_positive()

        return {}

    def get_constraints(self, verifier, V, Vdot):
        """
        :param verifier: verifier object
        :param V: SMT formula of Lyapunov Function
        :param Vdot: SMT formula of Lyapunov lie derivative
        :return: tuple of dictionaries of lyapunov conditons
        """
        _Or = verifier.solver_fncts()["Or"]
        _And = verifier.solver_fncts()["And"]
        _Not = verifier.solver_fncts()["Not"]

        lyap_negated = _Or(V <= 0, Vdot >= 0)

        not_origin = _Not(_And(*[xi == 0 for xi in verifier.xs]))
        lyap_negated = _And(lyap_negated, not_origin)
        lyap_condition = _And(self.domain, lyap_negated)
        for cs in ({fs.XD: lyap_condition},):
            yield cs

    def estimate_beta(self, net):
        # This function is unused I think
        try:
            border_D = self.D[fs.XD].sample_border(300)
            beta, _ = net.compute_minimum(border_D)
        except NotImplementedError:
            beta = self.D[fs.XD].generate_data(300)
        return beta


class NonPoly0(fs.control.DynamicalModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return [-x + x * y, -y]

    def f_smt(self, v):
        x, y = v
        return [-x + x * y, -y]


def test_lnn():
    system = NonPoly0
    X = fs.domains.Torus([0, 0], 1, 0.01)
    domain = {fs.XD: X}
    data = {fs.XD: X._generate_data(1000)}

    # define NN parameters
    activations = [fs.ActivationType.SQUARE]
    n_hidden_neurons = [6] * len(activations)

    ###
    #
    ###
    opts = fs.CegisConfig(
        SYSTEM=system,
        DOMAINS=domain,
        DATA=data,
        N_VARS=system.n_vars,
        CERTIFICATE=fs.CertificateType.CUSTOM,
        TIME_DOMAIN=fs.TimeDomain.CONTINUOUS,
        VERIFIER=fs.VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        CEGIS_MAX_ITERS=25,
        CUSTOM_CERTIFICATE=CustomLyapunov,
    )
    fs.synthesise(opts)


if __name__ == "__main__":
    # args = main.parse_benchmark_args()
    test_lnn()
