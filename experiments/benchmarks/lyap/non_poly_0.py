# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable

# from experiments.benchmarks import models
import fossil


class NonPoly0(fossil.control.DynamicalModel):
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return [-x + x * y, -y]

    def f_smt(self, v):
        x, y = v
        return [-x + x * y, -y]


def test_lnn():
    system = NonPoly0
    X = fossil.domains.Torus([0, 0], 1, 0.01)
    domain = {fossil.XD: X}
    data = {fossil.XD: X._generate_data(1000)}

    # define NN parameters
    activations = [fossil.ActivationType.SQUARE]
    n_hidden_neurons = [6] * len(activations)

    ###
    #
    ###
    opts = fossil.CegisConfig(
        SYSTEM=system,
        DOMAINS=domain,
        DATA=data,
        N_VARS=system.n_vars,
        CERTIFICATE=fossil.CertificateType.LYAPUNOV,
        TIME_DOMAIN=fossil.TimeDomain.CONTINUOUS,
        VERIFIER=fossil.VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        LLO=True,
        CEGIS_MAX_ITERS=25,
    )
    fossil.synthesise(opts)


if __name__ == "__main__":
    # args = main.parse_benchmark_args()
    test_lnn()
