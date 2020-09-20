from functools import partial

import torch
import numpy as np
import timeit

from src.shared.cegis_values import CegisConfig, CegisStateKeys, CegisComponentsState
from src.shared.consts import VerifierType, LearnerType
from src.barrier.utils import get_symbolic_formula, print_section, compute_trajectory
from src.barrier.net import NN
from src.shared.sympy_converter import *
from src.barrier.drealverifier import DRealVerifier


class Cegis:
    # todo: set params for NN and avoid useless definitions
    def __init__(self, n_vars, learner_type, verifier_type, active, system, n_hidden_neurons, **kw):
        self.sp_simplify = kw.get(CegisConfig.SP_SIMPLIFY.k, CegisConfig.SP_SIMPLIFY.v)
        self.sp_handle = kw.get(CegisConfig.SP_HANDLE.k, CegisConfig.SP_HANDLE.v)
        self.sb = kw.get(CegisConfig.SYMMETRIC_BELT.k, CegisConfig.SYMMETRIC_BELT.v)

        self.n = n_vars
        self.learner_type = learner_type
        self.verifier_type = verifier_type
        self.activ = active
        self.h = n_hidden_neurons
        self.max_cegis_iter = kw.get(CegisConfig.CEGIS_MAX_ITERS.k, CegisConfig.CEGIS_MAX_ITERS.v)
        self.max_cegis_time = kw.get(CegisConfig.CEGIS_MAX_TIME_S.k, CegisConfig.CEGIS_MAX_TIME_S.v)

        # batch init
        self.batch_size = kw.get(CegisConfig.BATCH_SIZE.k, CegisConfig.BATCH_SIZE.v)
        self.learning_rate = kw.get(CegisConfig.LEARNING_RATE.k, CegisConfig.LEARNING_RATE.v)

        self._assert_state()

        if verifier_type == VerifierType.Z3:
            verifier_class = Z3Verifier
        elif verifier_type == VerifierType.DREAL:
            verifier_class = DRealVerifier

        self.x = verifier_class.new_vars(self.n)
        self.x_map = {str(x): x for x in self.x}

        self.f, self.f_whole_domain, self.f_initial_state, self.f_unsafe_state, self.S_d, self.S_i, self.S_u, vars_bounds \
            = system(verifier_class.solver_fncts())
        self.domain = self.f_whole_domain(verifier_class.solver_fncts(), self.x)
        self.initial_s = self.f_initial_state(verifier_class.solver_fncts(), self.x)
        self.unsafe = self.f_unsafe_state(verifier_class.solver_fncts(), self.x)

        self.verifier = verifier_class(self.n, self.domain, self.initial_s, self.unsafe, vars_bounds, self.x)

        self.xdot = self.f(self.verifier.solver_fncts(), self.x)
        self.x = np.matrix(self.x).T
        self.xdot = np.matrix(self.xdot).T

        if learner_type == LearnerType.NN:
            self.learner = NN(n_vars, *n_hidden_neurons, activate=self.activ, bias=True, symmetric_belt=self.sb)
            self.optimizer = torch.optim.AdamW(self.learner.parameters(), lr=self.learning_rate)

        self.f_verifier = partial(self.f, self.verifier.solver_fncts())
        self.f_learner = partial(self.f, self.learner.learner_fncts())

        if self.sp_handle:
            self.x_sympy = [sp.Symbol('x%d' % i, real=True) for i in range(self.n)]
            self.xdot_s = self.f({'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp, }, self.x_sympy)
            self.x_sympy, self.xdot_s = np.matrix(self.x_sympy).T, np.matrix(self.xdot_s).T

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

        learner_to_next_component_inputs = {
            CegisStateKeys.x_v_map: self.x_map,
            CegisStateKeys.x_v: self.x,
            CegisStateKeys.x_v_dot: self.xdot,
            CegisStateKeys.x_sympy: self.x_sympy,
            CegisStateKeys.x_dot_sympy: self.xdot_s,
            CegisStateKeys.sp_simplify: self.sp_simplify,
            CegisStateKeys.sp_handle: self.sp_handle,
        }

        components = [
            {
                CegisComponentsState.name: 'learner',
                CegisComponentsState.instance: self.learner,
                CegisComponentsState.to_next_component: lambda _outputs, next_component, **kw:
                    self.learner.to_next_component(self.learner, next_component, **{
                        **learner_to_next_component_inputs, **kw
                    }),
            },
            {
                CegisComponentsState.name: 'verifier',
                CegisComponentsState.instance: self.verifier,
                CegisComponentsState.to_next_component: lambda _outputs, **kw: kw,
            },
        ]

        state = {
            CegisStateKeys.optimizer: self.optimizer,
            CegisStateKeys.S: S,
            CegisStateKeys.S_dot: Sdot,
            CegisStateKeys.B: None,
            CegisStateKeys.B_dot: None,
        }

        while not stop:
            for component_idx in range(len(components)):
                component = components[component_idx]
                next_component = components[(component_idx + 1) % len(components)]

                print_section(component[CegisComponentsState.name], iters)
                outputs = self.learner.get(**state)

                state = {**state, **outputs}

                print_section('Outputs', state)

                state = {**state,
                         **(component[CegisComponentsState.to_next_component]
                                (outputs, next_component[CegisComponentsState.instance], **state))}

                if state[CegisStateKeys.found]:
                    break

            if self.max_cegis_iter == iters or timeit.default_timer() - start > self.max_cegis_time:
                print('Out of Cegis resources: iters=%d elapsed time=%ss' % (iters, timeit.default_timer() - start))
                stop = True

            if state[CegisStateKeys.found]:
                print('Certified!')
                stop = True
            else:
                iters += 1
                S, Sdot = self.add_ces_to_data(S, Sdot, state[CegisStateKeys.cex])

                # compute climbing towards Bdot
                S, Sdot = self.trajectoriser(S, Sdot, state[CegisStateKeys.cex])

        print('Learner times: {}'.format(self.learner.get_timer()))
        print('Verifier times: {}'.format(self.verifier.get_timer()))
        return self.learner, state[CegisStateKeys.found], iters

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
                S[idx] = torch.cat([S[idx], ces[idx]], dim=0)
                Sdot[idx] = torch.cat([Sdot[idx], torch.stack(list(map(torch.tensor, map(self.f_learner, ces[idx]))))], dim=0)
        return S, Sdot

    def trajectoriser(self, S, Sdot, ces):
        ce = ces[0]
        if len(ce) > 0:
            point = ce[-1]
            point.requires_grad = True
            trajectory = compute_trajectory(self.learner, point, self.f_learner)
            S, Sdot = self.add_ces_to_data(S, Sdot, [torch.stack(trajectory), [], []])

        return S, Sdot

    def _assert_state(self):
        assert self.verifier_type in [VerifierType.Z3, VerifierType.DREAL]
        assert self.learner_type in [LearnerType.NN]
        assert self.batch_size > 0
        assert self.learning_rate > 0
        assert self.max_cegis_time > 0

