from functools import partial

import torch
import numpy as np
import timeit

from src.shared.consts import VerifierType, LearnerType
from src.barrier.utils import get_symbolic_formula, print_section, compute_trajectory
from src.barrier.net import NN
from src.shared.sympy_converter import *
from src.barrier.drealverifier import DRealVerifier


class Cegis:
    # todo: set params for NN and avoid useless definitions
    def __init__(self, n_vars, learner_type, verifier_type, active, system, n_hidden_neurons, **kw):
        self.sp_simplify = kw.get('sp_simplify', True)
        self.sp_handle = kw.get('sp_handle', True)
        self.sb = kw.get('symmetric_belt', False)

        self.n = n_vars
        self.learner_type = learner_type
        self.activ = active
        self.h = n_hidden_neurons
        self.max_cegis_iter = kw.get('cegis_iters', 200)
        self.max_cegis_time = kw.get('cegis_time', math.inf)

        # batch init
        self.batch_size = 100
        self.learning_rate = .1

        if verifier_type == VerifierType.Z3:
            verifier = Z3Verifier
        elif verifier_type == VerifierType.DREAL:
            verifier = DRealVerifier
        else:
            raise ValueError('No verifier of type {}'.format(verifier_type))

        self.x = verifier.new_vars(self.n)
        self.x_map = {str(x): x for x in self.x}

        self.f, self.f_whole_domain, self.f_initial_state, self.f_unsafe_state, self.S_d, self.S_i, self.S_u, vars_bounds \
            = system(verifier.solver_fncts())
        self.domain = self.f_whole_domain(verifier.solver_fncts(), self.x)
        self.initial_s = self.f_initial_state(verifier.solver_fncts(), self.x)
        self.unsafe = self.f_unsafe_state(verifier.solver_fncts(), self.x)

        self.verifier = verifier(self.n, self.domain, self.initial_s, self.unsafe, vars_bounds, self.x)

        self.xdot = self.f(self.verifier.solver_fncts(), self.x)
        self.x = np.matrix(self.x).T
        self.xdot = np.matrix(self.xdot).T

        if learner_type == LearnerType.NN:
            self.learner = NN(n_vars, *n_hidden_neurons, activate=self.activ, bias=True, symmetric_belt=self.sb)
            self.optimizer = torch.optim.AdamW(self.learner.parameters(), lr=self.learning_rate)
        else:
            raise ValueError('No learner of type {}'.format(learner_type))

        self.f_verifier = partial(self.f, self.verifier.solver_fncts())
        self.f_learner = partial(self.f, self.learner.learner_fncts())

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
        stop, found = False, False
        start = timeit.default_timer()
        if self.sp_handle:
            x_sympy = [sp.Symbol('x%d' % i, real=True) for i in range(self.n)]
            xdot_s = self.f({'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp, }, x_sympy)
            x_sympy, xdot_s = np.matrix(x_sympy).T, np.matrix(xdot_s).T

        while not stop:
            print_section('Learning', iters)
            self.learner.learn(self.optimizer, S, Sdot)
            if not self.sp_handle:  # z3 does all the handling
                B, Bdot = get_symbolic_formula(self.learner, self.x, self.xdot)
                if isinstance(B, z3.ArithRef):
                    B, Bdot = z3.simplify(B), z3.simplify(Bdot)
                B_s, Bdot_s = B, Bdot
            else:
                B_s, Bdot_s = get_symbolic_formula(self.learner, x_sympy, xdot_s)

            if self.sp_simplify:
                B_s = sp.simplify(B_s)
                Bdot_s = sp.simplify(Bdot_s)
            if self.sp_handle:
                B = sympy_converter(B_s, target=self.verifier, var_map=self.x_map)
                Bdot = sympy_converter(Bdot_s, target=self.verifier, var_map=self.x_map)

            print_section('Candidate', iters)
            print(f'B: {B_s}')
            print(f'Bdot: {Bdot_s}')

            print_section('Verification', iters)
            found, ces = self.verifier.verify(B, Bdot)

            if self.max_cegis_iter == iters or timeit.default_timer() - start > self.max_cegis_time:
                print('Out of Cegis resources: iters=%d elapsed time=%ss' % (iters, timeit.default_timer() - start))
                stop = True

            if found:
                print('Certified!')
                print(f'B: {B}')
                print(f'Bdot: {Bdot}')
                stop = True
            else:
                iters += 1
                S, Sdot = self.add_ces_to_data(S, Sdot, ces)

                # compute climbing towards Bdot
                S, Sdot = self.trajectoriser(S, Sdot, ces)

        print('Learner times: {}'.format(self.learner.get_timer()))
        print('Verifier times: {}'.format(self.verifier.get_timer()))
        return self.learner, found, iters

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
