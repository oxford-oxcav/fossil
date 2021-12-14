# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
from functools import partial

from src.shared.components.primer import Primer
from src.shared.system import NonlinearSystem
from experiments.benchmarks.domain_fcns import * 
from src.shared.components.cegis import Cegis
from src.shared.consts import TimeDomain, CertificateType
from src.shared.activations import ActivationType
from src.shared.utils import FailedSynthesis
from src.shared.cegis_values import CegisStateKeys, CegisConfig


class PrimerBarrier(Primer):
    def __init__(self, f, xd, xi, xu, **kw):
        self.cegis_parameters = kw.get(CegisConfig.CEGIS_PARAMETERS.k, CegisConfig.CEGIS_PARAMETERS.v)
        if not callable(f):
            # f is a list of sympy dynamics and Primer must generate the system and domains
            self.dynamics = NonlinearSystem(f, False)
            self.dimension = self.dynamics.dimension
            self.sxd, self.sxi, self.sxu = xd, xi, xu
        else:
            # f is directly a 'system' function of the form in the benchmarks
            self.dynamics = f
            self.dimension = self.cegis_parameters.get(CegisConfig.N_VARS.k, CegisConfig.N_VARS.v)
            if self.dimension == 0:
                raise TypeError('CEGIS Parameter N_VARS must be passed if f is in the form of a python function.')

        self.seed_and_speed_handle = self.cegis_parameters.get(CegisConfig.SEED_AND_SPEED.k, CegisConfig.SEED_AND_SPEED.v)
        self.batch_size = self.cegis_parameters.get(CegisConfig.BATCH_SIZE.k, CegisConfig.BATCH_SIZE.v)

    def get(self):
        """
        :return B_n: numerical form of Barrier function
        :return B_v: symbolic form of Barrier function
        """
        if self.seed_and_speed_handle:
            state, f_learner = self.seed_and_speed()
        else:
            state, f_learner = self.run_cegis()
        
        if not state[CegisStateKeys.found]:
            raise FailedSynthesis('Function could not be synthesised.')

        learner = state[CegisStateKeys.net]

        def B_n(x):
            """
            :param x: iterable of shape (N, dimension) (torch tensor recommended)
            :return B, Bdot:
            """
            if isinstance(x, torch.Tensor):
                x = x.reshape(-1, self.dimension)
            else:
                x = torch.tensor(x).reshape(-1, self.dimension)
            xdot = list(map(torch.tensor, map(f_learner, x)))
            xdot = torch.stack(xdot)
            B, Bdot, _ = learner.numerical_net(x, xdot)
            return B, Bdot

        B_v = state[CegisStateKeys.V]
        return B_n, B_v

    def run_cegis(self):
        """
        :return state: dict,  cegis state dictionary
        :return f_learner: function that evaluates xdot of system
        """
        if callable(self.dynamics):
            system = partial(self.dynamics, self.batch_size)
        else:
            system = self.system

        activations = self.cegis_parameters.get(CegisConfig.ACTIVATION.k, CegisConfig.ACTIVATION.v)
        neurons = self.cegis_parameters.get(CegisConfig.N_HIDDEN_NEURONS.k, CegisConfig.N_HIDDEN_NEURONS.v)
        verifier = self.cegis_parameters.get(CegisConfig.VERIFIER.k, CegisConfig.VERIFIER.v)

        self.check_verifier(verifier)
        params = {CegisConfig.N_VARS.k:self.dimension, 
                  CegisConfig.SYSTEM.k: system,
                  CegisConfig.ACTIVATION.k: activations,
                  CegisConfig.N_HIDDEN_NEURONS.k: neurons,
                  CegisConfig.VERIFIER.k: verifier, 
                  CegisConfig.TIME_DOMAIN.k: TimeDomain.CONTINUOUS,
                  CegisConfig.CERTIFICATE.k: CertificateType.LYAPUNOV
                  }

        self.cegis_parameters.update(params)
        c = Cegis(**self.cegis_parameters)

        state, x, f_learner, iters = c.solve()
        return state, f_learner

    def system(self, functions, inner=0.0, outer=10.0):
        _And = functions["And"]
        _Or  = functions["Or"]
        bounds = inf_bounds_n(self.dimension)

        def f(_, v):
            return self.dynamics.evaluate_f(v)

        def XD(_, v):
            return self.sxd.generate_domain(v, _And)

        def XI(_, v):
            return self.sxi.generate_domain(v, _And)

        def XU(_, v):
            return self.sxu.generate_domain(v, _And)

        def SD():
            return self.sxd.generate_data(self.batch_size)

        def SI():
            return self.sxi.generate_data(self.batch_size)

        def SU():
            return self.sxu.generate_data(self.batch_size)

        return f, XD, XI, XU, SD(), SI(), SU(), bounds
