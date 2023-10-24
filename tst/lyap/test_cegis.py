# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import torch
import timeit
from fossil.cegis import Cegis
from experiments.benchmarks.benchmarks_lyap import *
from fossil.consts import *
from z3 import *
import fossil.translator as translator
from fossil.consts import CegisStateKeys


def zero_in_zero(learner):
    v_zero = learner.forward(
        torch.zeros(1, learner.input_size).reshape(1, learner.input_size),
    )
    return v_zero == 0.0


def positive_definite(learner, S_d, Sdot):
    v, _, _ = learner.get_all(S_d, Sdot)
    return all(v >= 0.0)


def negative_definite_lie_derivative(learner, S, Sdot):
    v, vdot, _ = learner.get_all(S, Sdot)
    # find points have vdot > 0
    if len(torch.nonzero(vdot > 0)) > 0:
        return False
    else:
        return True


class test_cegis(unittest.TestCase):
    def assertNumericallyLyapunov(self, model, S_d, Sdot):
        self.assertTrue(zero_in_zero(model), "model is not zero in zero")
        self.assertTrue(
            positive_definite(model, S_d, Sdot), "model is not positive definite"
        )
        self.assertTrue(
            negative_definite_lie_derivative(model, S_d, Sdot),
            "model lie derivative is not negative definite",
        )

    def assertLyapunovOverPositiveOrthant(self, f, c, domain):
        domain = domain["lie"].generate_domain(list(c.x_map.values()))
        tr = translator.TranslatorCT(c.x, c.xdot, 3, CegisConfig())
        res = tr.get(**{"net": c.learner})
        V, Vdot = res[CegisStateKeys.V], res[CegisStateKeys.V_dot]

        s = Solver()
        s.add(Vdot > 0)
        s.add(domain)
        res = s.check()
        if res == z3.sat:
            model = "{}".format(s.model())
        else:
            model = ""
        self.assertEqual(
            res, z3.unsat, "Formally not lyapunov. Here is a cex : {}".format(model)
        )

        Sdot = f()(c.S["lie"])
        S = c.S["lie"]
        self.assertNumericallyLyapunov(c.learner, S, Sdot)

    def test_poly_2(self):
        torch.manual_seed(167)

        outer = 10.0
        inner = 0.01
        batch_size = 500

        f = models.Poly2

        XD = Torus([0.0, 0.0], outer, inner)

        domains = {
            certificate.XD: XD,
        }

        data = {
            certificate.XD: XD._generate_data(batch_size),
        }

        n_vars = 2

        # define NN parameters
        activations = [ActivationType.SQUARE]
        n_hidden_neurons = [10] * len(activations)

        opts = CegisConfig(
            N_VARS=n_vars,
            DATA=data,
            DOMAINS=domains,
            TIME_DOMAIN=TimeDomain.CONTINUOUS,
            CERTIFICATE=CertificateType.LYAPUNOV,
            VERIFIER=VerifierType.Z3,
            ACTIVATION=activations,
            SYSTEM=f,
            N_HIDDEN_NEURONS=n_hidden_neurons,
            LLO=True,
        )

        start = timeit.default_timer()
        c = Cegis(opts)
        c.solve()
        stop = timeit.default_timer()

        self.assertLyapunovOverPositiveOrthant(f, c, domains)

    def test_non_poly_0(self):
        torch.manual_seed(167)
        n_vars = 2
        f = models.NonPoly0
        domain = Torus([0, 0], 1, 0.01)

        domains = {certificate.XD: domain}
        data = {certificate.XD: domain._generate_data(1000)}

        # define domain constraints

        # define NN parameters
        activations = [ActivationType.SQUARE]
        n_hidden_neurons = [2] * len(activations)

        opts = CegisConfig(
            N_VARS=n_vars,
            DATA=data,
            DOMAINS=domains,
            TIME_DOMAIN=TimeDomain.CONTINUOUS,
            CERTIFICATE=CertificateType.LYAPUNOV,
            VERIFIER=VerifierType.Z3,
            ACTIVATION=activations,
            SYSTEM=f,
            N_HIDDEN_NEURONS=n_hidden_neurons,
            LLO=True,
        )

        start = timeit.default_timer()
        c = Cegis(opts)
        c.solve()
        stop = timeit.default_timer()

        self.assertLyapunovOverPositiveOrthant(f, c, domains)

    def test_non_poly_1(self):
        torch.manual_seed(167)
        outer = 10.0
        batch_size = 500

        f = models.NonPoly1

        XD = PositiveOrthantSphere([0.0, 0.0], outer)

        domains = {
            certificate.XD: XD,
        }

        data = {
            certificate.XD: XD._generate_data(batch_size),
        }
        n_vars = 2

        # define NN parameters
        activations = [ActivationType.LINEAR, ActivationType.SQUARE]
        n_hidden_neurons = [20] * len(activations)

        opts = CegisConfig(
            N_VARS=n_vars,
            DATA=data,
            DOMAINS=domains,
            TIME_DOMAIN=TimeDomain.CONTINUOUS,
            CERTIFICATE=CertificateType.LYAPUNOV,
            VERIFIER=VerifierType.Z3,
            ACTIVATION=activations,
            SYSTEM=f,
            N_HIDDEN_NEURONS=n_hidden_neurons,
            LLO=True,
        )

        start = timeit.default_timer()
        c = Cegis(opts)
        c.solve()
        stop = timeit.default_timer()

        self.assertLyapunovOverPositiveOrthant(f, c, domains)

    def test_non_poly_2(self):
        torch.manual_seed(167)
        n_vars = 3
        outer = 10.0
        batch_size = 500

        f = models.NonPoly2

        XD = PositiveOrthantSphere([0.0, 0.0, 0.0], outer)

        domains = {
            certificate.XD: XD,
        }

        data = {
            certificate.XD: XD._generate_data(batch_size),
        }

        # define NN parameters
        activations = [ActivationType.LINEAR, ActivationType.SQUARE]
        n_hidden_neurons = [10] * len(activations)

        start = timeit.default_timer()

        opts = CegisConfig(
            N_VARS=n_vars,
            DATA=data,
            DOMAINS=domains,
            TIME_DOMAIN=TimeDomain.CONTINUOUS,
            CERTIFICATE=CertificateType.LYAPUNOV,
            VERIFIER=VerifierType.Z3,
            ACTIVATION=activations,
            SYSTEM=f,
            N_HIDDEN_NEURONS=n_hidden_neurons,
            LLO=True,
        )

        start = timeit.default_timer()
        c = Cegis(opts)
        c.solve()
        stop = timeit.default_timer()

        self.assertLyapunovOverPositiveOrthant(f, c, domains)


if __name__ == "__main__":
    unittest.main()
