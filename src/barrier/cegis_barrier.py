# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
from functools import partial

import torch
import numpy as np
import timeit

from src.shared.cegis_values import CegisConfig, CegisStateKeys, CegisComponentsState
from src.shared.consts import VerifierType, LearnerType, ConsolidatorType, TranslatorType
from src.barrier.utils import print_section, compute_trajectory, vprint
from src.barrier.net import NN
from src.shared.sympy_converter import *
from src.barrier.drealverifier import DRealVerifier
from src.shared.components.consolidator import Consolidator
from src.shared.components.translator_continuous import TranslatorContinuous


class Cegis:
    # todo: set params for NN and avoid useless definitions
    def __init__(self, **kw):
        self.n = kw[CegisConfig.N_VARS.k]
        # components types
        self.learner_type = kw[CegisConfig.LEARNER.k]
        self.verifier_type = kw[CegisConfig.VERIFIER.k]
        self.consolidator_type = kw[CegisConfig.CONSOLIDATOR.k]
        self.translator_type = kw[CegisConfig.TRANSLATOR.k]
        # benchmark options
        self.activ = kw[CegisConfig.ACTIVATION.k]
        self.system = kw[CegisConfig.SYSTEM.k]
        self.h = kw[CegisConfig.N_HIDDEN_NEURONS.k]
        self.sp_simplify = kw.get(CegisConfig.SP_SIMPLIFY.k, CegisConfig.SP_SIMPLIFY.v)
        self.sp_handle = kw.get(CegisConfig.SP_HANDLE.k, CegisConfig.SP_HANDLE.v)
        self.sb = kw.get(CegisConfig.SYMMETRIC_BELT.k, CegisConfig.SYMMETRIC_BELT.v)
        self.eq = kw.get(CegisConfig.EQUILIBRIUM.k, CegisConfig.EQUILIBRIUM.v[0](self.n))
        self.rounding = kw.get(CegisConfig.ROUNDING.k, CegisConfig.ROUNDING.v)
        self.fcts = kw.get(CegisConfig.FACTORS.k, CegisConfig.FACTORS.v)
        # other opts
        self.max_cegis_iter = kw.get(CegisConfig.CEGIS_MAX_ITERS.k, CegisConfig.CEGIS_MAX_ITERS.v)
        self.max_cegis_time = kw.get(CegisConfig.CEGIS_MAX_TIME_S.k, CegisConfig.CEGIS_MAX_TIME_S.v)
        self.verbose = kw.get(CegisConfig.VERBOSE.k, CegisConfig.VERBOSE.v)
        # batch init
        self.batch_size = kw.get(CegisConfig.BATCH_SIZE.k, CegisConfig.BATCH_SIZE.v)
        self.learning_rate = kw.get(CegisConfig.LEARNING_RATE.k, CegisConfig.LEARNING_RATE.v)

        self._assert_state()

        if self.verifier_type == VerifierType.Z3:
            verifier_class = Z3Verifier
        elif self.verifier_type == VerifierType.DREAL:
            verifier_class = DRealVerifier

        self.x = verifier_class.new_vars(self.n)
        self.x_map = {str(x): x for x in self.x}

        self.f, self.f_whole_domain, self.f_initial_state, self.f_unsafe_state, self.S_d, self.S_i, self.S_u, vars_bounds \
            = self.system(verifier_class.solver_fncts())
        self.domain = self.f_whole_domain(verifier_class.solver_fncts(), self.x)
        self.initial_s = self.f_initial_state(verifier_class.solver_fncts(), self.x)
        self.unsafe = self.f_unsafe_state(verifier_class.solver_fncts(), self.x)

        self.verifier = verifier_class(self.n, self.domain, self.initial_s, self.unsafe, vars_bounds, self.x, **kw)

        self.xdot = self.f(self.verifier.solver_fncts(), self.x)
        self.x = np.array(self.x).reshape(-1, 1)
        self.xdot = np.array(self.xdot).reshape(-1, 1)

        if self.learner_type == LearnerType.CONTINUOUS:
            self.learner = NN(self.n, *self.h, activate=self.activ, bias=True, **kw)
            self.optimizer = torch.optim.AdamW(self.learner.parameters(), lr=self.learning_rate)

        self.f_verifier = partial(self.f, self.verifier.solver_fncts())
        self.f_learner = partial(self.f, self.learner.learner_fncts())

        if self.sp_handle:
            self.x = [sp.Symbol('x%d' % i, real=True) for i in range(self.n)]
            self.xdot = self.f({'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp}, self.x)
            self.x_map = {**self.x_map, **self.verifier.solver_fncts()}
            self.x, self.xdot = np.array(self.x).reshape(-1,1), np.array(self.xdot).reshape(-1,1)
        else:
            self.x_sympy, self.xdot_s = None, None

        if self.consolidator_type == ConsolidatorType.DEFAULT:
            self.consolidator = Consolidator(self.f_learner)
        else:
            TypeError('Not Implemented Consolidator')
        if self.translator_type == TranslatorType.DEFAULT:
            self.translator = Translator(self.learner, self.x, self.xdot, self.eq, self.rounding, **kw)
        else:
            TypeError('Not Implemented Translator')

        self._result = None

    def solve(self):
        """
        :return:
        """
        # order of elements is: domain, init, unsafe
        S_d, S_i, S_u = self.S_d, self.S_i, self.S_u
        Sdot_d = list(map(torch.tensor, map(self.f_learner, S_d)))
        Sdot_d = torch.stack(Sdot_d)

        Sdot_i, Sdot_u = list(map(torch.tensor, map(self.f_learner, S_i))), list(map(torch.tensor, map(self.f_learner, S_u)))
        Sdot_i, Sdot_u = torch.stack(Sdot_i), torch.stack(Sdot_u)

        S, Sdot = [S_d, S_i, S_u], [Sdot_d, Sdot_i, Sdot_u]

        # the CEGIS loop
        iters = 0
        stop = False
        start = timeit.default_timer()

        components = [
            {
                CegisComponentsState.name: 'learner',
                CegisComponentsState.instance: self.learner,
                CegisComponentsState.to_next_component: lambda _outputs, next_component, **kw: kw,
            },
            {
                CegisComponentsState.name: 'translator',
                CegisComponentsState.instance: self.translator,
                CegisComponentsState.to_next_component: lambda _outputs, next_component, **kw: kw,
            },
            {
                CegisComponentsState.name: 'verifier',
                CegisComponentsState.instance: self.verifier,
                CegisComponentsState.to_next_component: lambda _outputs, next_component, **kw: kw,
            },
            {
                CegisComponentsState.name: 'consolidator',
                CegisComponentsState.instance: self.consolidator,
                CegisComponentsState.to_next_component: lambda _outputs, next_component, **kw: kw
            }
        ]

        state = {
            CegisStateKeys.net: self.learner,
            CegisStateKeys.optimizer: self.optimizer,
            CegisStateKeys.sp_handle:self.sp_handle,
            CegisStateKeys.S: S,
            CegisStateKeys.S_dot: Sdot,
            CegisStateKeys.factors: self.fcts, # default in consolidator
            CegisStateKeys.B: None,
            CegisStateKeys.B_dot: None,
            CegisStateKeys.x_v_map: self.x_map,
            CegisStateKeys.verifier_fun: self.f_verifier,
            CegisStateKeys.found: False,
            CegisStateKeys.verification_timed_out: False,
            CegisStateKeys.cex: None,
            CegisStateKeys.trajectory: None,
        }

        # reset timers
        self.learner.get_timer().reset()
        self.translator.get_timer().reset()
        self.verifier.get_timer().reset()
        self.consolidator.get_timer().reset()

        while not stop:
            for component_idx in range(len(components)):
                component = components[component_idx]
                next_component = components[(component_idx + 1) % len(components)]

                if self.verbose:
                    print_section(component[CegisComponentsState.name], iters)
                outputs = component[CegisComponentsState.instance].get(**state)

                state = {**state, **outputs}

                state = {**state,
                         **(component[CegisComponentsState.to_next_component]
                                (outputs, next_component[CegisComponentsState.instance], **state))}

                if state[CegisStateKeys.found] and component_idx == len(components) - 1:
                    print('Certified!')
                    stop = True
                if state[CegisStateKeys.verification_timed_out]:
                    print('Verification Timed Out')
                    stop = True

            if (self.max_cegis_iter == iters or timeit.default_timer() - start > self.max_cegis_time) and not state[CegisStateKeys.found]:
                print('Out of Cegis resources: iters=%d elapsed time=%ss' % (iters, timeit.default_timer() - start))
                stop = True

            iters += 1
            if not (state[CegisStateKeys.found] or state[CegisStateKeys.verification_timed_out]):
                # add trajectory to the first set of cex
                if len(state[CegisStateKeys.cex][0]) > 0:
                    state[CegisStateKeys.cex][0] = torch.cat([state[CegisStateKeys.cex][0],
                                                             state[CegisStateKeys.trajectory]])
                state[CegisStateKeys.S], state[CegisStateKeys.S_dot] = \
                    self.add_ces_to_data(state[CegisStateKeys.S], state[CegisStateKeys.S_dot],
                                         state[CegisStateKeys.cex])

        state[CegisStateKeys.components_times] = [
            self.learner.get_timer().sum, self.translator.get_timer().sum,
            self.verifier.get_timer().sum, self.consolidator.get_timer().sum
        ]

        vprint(['Learner times: {}'.format(self.learner.get_timer())], self.verbose)
        vprint(['Translator times: {}'.format(self.translator.get_timer())], self.verbose)
        vprint(['Verifier times: {}'.format(self.verifier.get_timer())], self.verbose)
        vprint(['Consolidator times: {}'.format(self.consolidator.get_timer())], self.verbose)

        self._result = state, self.x, self.f_learner, iters
        return self._result

    @property
    def result(self):
        return self._result

    def add_ces_to_data(self, S, Sdot, ces):
        """
        :param S: torch tensor
        :param Sdot: torch tensor
        :param ces: list of ctx
        :return:
                S: torch tensor, added new ctx
                Sdot torch tensor, added  f(new_ctx)
        """
        for idx in range(3):
            if len(ces[idx]) != 0:
                S[idx] = torch.cat([S[idx], ces[idx]], dim=0).detach()
                # Sdot[idx] = torch.stack(self.f_learner(S[idx].T)).T
                Sdot[idx] = list(map(torch.tensor, map(self.f_learner, S[idx])))
                Sdot[idx] = torch.stack(Sdot[idx])

                # S[idx] = torch.cat([S[idx], ces[idx]], dim=0)
                # Sdot[idx] = torch.cat([Sdot[idx],
                #                       torch.stack(list(map(torch.tensor,
                #                                   map(self.f_learner, ces[idx]))))], dim=0)
        return S, Sdot

    def consolidator_method(self, S, Sdot, ces):
        ce = ces[0]
        if len(ce) > 0:
            point = ce[-1]
            point.requires_grad = True
            trajectory = compute_trajectory(self.learner, point, self.f_learner)
            S, Sdot = self.add_ces_to_data(S, Sdot, [torch.stack(trajectory), [], []])

        return S, Sdot

    def _assert_state(self):
        assert self.verifier_type in [VerifierType.Z3, VerifierType.DREAL]
        assert self.learner_type in [LearnerType.CONTINUOUS]
        assert self.batch_size > 0
        assert self.learning_rate > 0
        assert self.max_cegis_time > 0

