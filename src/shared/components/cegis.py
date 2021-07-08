# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
import torch
import numpy as np
import sympy as sp
import logging

from src.shared.cegis_values import CegisStateKeys, CegisConfig, CegisComponentsState
from src.verifier.drealverifier import DRealVerifier
from src.shared.consts import LearnerType, VerifierType, ConsolidatorType, TranslatorType, CertificateType
from src.verifier.z3verifier import Z3Verifier
from src.shared.components.consolidator import Consolidator
from src.certificate.lyapunov_certificate import LyapunovCertificate
from src.certificate.barrier_certificate import BarrierCertificate
from src.translator.translator_continuous import TranslatorContinuous
from src.translator.translator_discrete import TranslatorDiscrete
from src.shared.utils import print_section, vprint, rotate
from src.learner.net_continuous import NNContinuous
from src.learner.net_discrete import NNDiscrete
from functools import partial

try:
    import dreal as dr
except Exception as e:
    logging.exception('Exception while importing dReal')


class Cegis:
    # todo: set params for NN and avoid useless definitions
    def __init__(self, **kw):
        self.n = kw[CegisConfig.N_VARS.k]
        # components type
        self.verifier_type = kw[CegisConfig.VERIFIER.k]
        self.certificate_type = kw.get(CegisConfig.CERTIFICATE.k)
        self.consolidator_type = kw.get(CegisConfig.CONSOLIDATOR.k, CegisConfig.CONSOLIDATOR.v)
        self.time_domain = kw.get(CegisConfig.TIME_DOMAIN.k, CegisConfig.TIME_DOMAIN.v)
        self.learner_type = LearnerType[self.time_domain.name]
        self.translator_type = TranslatorType[self.time_domain.name]
        # benchmark opts
        self.inner = kw.get(CegisConfig.INNER_RADIUS.k, CegisConfig.INNER_RADIUS.v)
        self.outer = kw.get(CegisConfig.OUTER_RADIUS.k, CegisConfig.OUTER_RADIUS.v)
        self.h = kw[CegisConfig.N_HIDDEN_NEURONS.k]
        self.activations = kw[CegisConfig.ACTIVATION.k]
        self.system = kw[CegisConfig.SYSTEM.k]
        self.sp_simplify = kw.get(CegisConfig.SP_SIMPLIFY.k, CegisConfig.SP_SIMPLIFY.v)
        self.sp_handle = kw.get(CegisConfig.SP_HANDLE.k, CegisConfig.SP_HANDLE.v)
        self.fcts = kw.get(CegisConfig.FACTORS.k, CegisConfig.FACTORS.v)
        self.eq = kw.get(CegisConfig.EQUILIBRIUM.k, CegisConfig.EQUILIBRIUM.v[0](self.n))
        self.llo = kw.get(CegisConfig.LLO.k, CegisConfig.LLO.v)
        self.rounding = kw.get(CegisConfig.ROUNDING.k, CegisConfig.ROUNDING.v)
        # other opts
        self.max_cegis_iter = kw.get(CegisConfig.CEGIS_MAX_ITERS.k, CegisConfig.CEGIS_MAX_ITERS.v)
        self.verbose = kw.get(CegisConfig.VERBOSE.k, CegisConfig.VERBOSE.v)
        # batch init
        self.learning_rate = kw.get(CegisConfig.LEARNING_RATE.k, CegisConfig.LEARNING_RATE.v)

        if self.verifier_type == VerifierType.Z3:
            verifier = Z3Verifier
        elif self.verifier_type == VerifierType.DREAL:
            verifier = DRealVerifier
        else:
            raise ValueError('No verifier of type {}'.format(self.verifier_type))

        self.x = verifier.new_vars(self.n)
        self.x_map = {str(x): x for x in self.x}

        self.f, self.f_domains, self.S, vars_bounds = \
            self.system(functions=verifier.solver_fncts(), inner=self.inner, outer=self.outer)
        # self.S_d = self.S_d.requires_grad_(True)

        # self.verifier = verifier(self.n, self.domain, self.initial_s, self.unsafe, vars_bounds, self.x)
        self.domains = [f_domain(verifier.solver_fncts(), self.x) for f_domain in self.f_domains]
        if self.certificate_type == CertificateType.LYAPUNOV:
            self.certificate = LyapunovCertificate(XD=self.domains[0], **kw)
            is_bias = False
        elif self.certificate_type == CertificateType.BARRIER:
            self.certificate = BarrierCertificate(XD=self.domains[0], XI=self.domains[1], XU=self.domains[2], **kw)
            is_bias = True
        else:
            raise ValueError('No certificate of type {}'.format(self.certificate_type))

        self.verifier = verifier(self.n, self.certificate.get_constraints, self.domains[0], vars_bounds, self.x, **kw)

        self.xdot = self.f(self.verifier.solver_fncts(), self.x)
        self.x = np.array(self.x).reshape(-1, 1)
        self.xdot = np.array(self.xdot).reshape(-1, 1)

        if self.learner_type == LearnerType.CONTINUOUS:
            self.learner = NNContinuous(self.n, self.certificate.learn, *self.h, bias=is_bias, activate=self.activations,
                              equilibria=self.eq, llo=self.llo, **kw)
            self.optimizer = torch.optim.AdamW(self.learner.parameters(), lr=self.learning_rate)
        elif self.learner_type == LearnerType.DISCRETE:
            self.learner = NNDiscrete(self.n, self.certificate.learn, *self.h, bias=False, activate=self.activations,
                              equilibria=self.eq, llo=self.llo, **kw)
            self.optimizer = torch.optim.AdamW(self.learner.parameters(), lr=self.learning_rate)

        else:
            raise ValueError('No learner of type {}'.format(self.learner_type))

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
        if self.translator_type == TranslatorType.CONTINUOUS:
            self.translator = TranslatorContinuous(self.learner, self.x, self.xdot, self.eq, self.rounding, **kw)
        elif self.translator_type == TranslatorType.DISCRETE:
            self.translator = TranslatorDiscrete(self.learner, self.x, self.xdot, self.eq, self.rounding, **kw)
            
        else:
            TypeError('Not Implemented Translator')

        self._result = None

    # the cegis loop
    # todo: fix return, fix map(f, S)
    def solve(self):

        # Sdot = self.f_learner(self.S_d.T)
        # needed to make hybrid work
        Sdot = [list(map(torch.tensor, map(self.f_learner, S))) for S in self.S ]
        S, Sdot = rotate(self.S, 1), rotate([torch.stack(sdot) for sdot in Sdot], 1) 

        if self.learner_type == LearnerType.CONTINUOUS:
            self.optimizer = torch.optim.AdamW(self.learner.parameters(), lr=self.learning_rate)

        stats = {}
        # the CEGIS loop
        iters = 0
        stop = False

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
            CegisStateKeys.S: S,
            CegisStateKeys.S_dot: Sdot,
            CegisStateKeys.factors: self.fcts,
            CegisStateKeys.sp_handle: self.sp_handle,
            CegisStateKeys.V: None,
            CegisStateKeys.V_dot: None,
            CegisStateKeys.x_v_map: self.x_map,
            CegisStateKeys.verifier_fun: self.f_verifier,
            CegisStateKeys.found: False,
            CegisStateKeys.verification_timed_out: False,
            CegisStateKeys.cex: None,
            CegisStateKeys.trajectory: None
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

                state = {**state, **(
                    component[CegisComponentsState.to_next_component](
                        outputs, next_component[CegisComponentsState.instance], **state
                    ))}

                if state[CegisStateKeys.found] and component_idx == len(components)-1:
                    print('Found a Lyapunov function')
                    stop = True
                if state[CegisStateKeys.verification_timed_out]:
                    print('Verification Timed Out')
                    stop = True

            if self.max_cegis_iter == iters and not state[CegisStateKeys.found]:
                print('Out of Cegis loops')
                stop = True

            iters += 1
            if not (state[CegisStateKeys.found] or state[CegisStateKeys.verification_timed_out]):
                self.learner.find_closest_unsat(state[CegisStateKeys.S], state[CegisStateKeys.S_dot])
                if len(state[CegisStateKeys.cex]) == 3 or len(state[CegisStateKeys.cex]) == 1:
                    state[CegisStateKeys.cex][-1] = torch.cat([state[CegisStateKeys.cex][-1],
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
        for idx in range(len(ces)):
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