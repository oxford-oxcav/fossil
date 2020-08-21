import torch
import numpy as np
import sympy as sp
from z3 import *
import logging
from shared.consts import LearnerType, VerifierType
from lyap.verifier.z3verifier import Z3Verifier
from lyap.verifier.drealverifier import DRealVerifier
from lyap.utils import get_symbolic_formula, print_section, compute_trajectory
from lyap.learner.net import NN
from functools import partial
from shared.sympy_converter import sympy_converter
try:
    import dreal as dr
except Exception as e:
    logging.exception('Exception while importing dReal')


class Cegis():
    # todo: set params for NN and avoid useless definitions
    # (n_vars, system, learner_type, activations, n_hidden_neurons, verifier_type, inner_radius, outer_radius,
    #               linear_factor=linear_factors)
    def __init__(self, n_vars, system, learner_type, activations, n_hidden_neurons,
                 verifier_type, inner_radius, outer_radius,
                 **kw):
        self.sp_simplify = kw.get('sp_simplify', True)
        self.sp_handle = kw.get('sp_handle', True)
        self.fcts = kw.get('factors', None)
        self.eq = kw.get('eq', None)

        self.n = n_vars
        self.learner_type = learner_type
        self.inner = inner_radius
        self.outer = outer_radius
        self.h = n_hidden_neurons
        self.max_cegis_iter = 5

        # batch init
        self.learning_rate = .05

        if verifier_type == VerifierType.Z3:
            verifier = Z3Verifier
        elif verifier_type == VerifierType.DREAL:
            verifier = DRealVerifier
        else:
            raise ValueError('No verifier of type {}'.format(verifier_type))

        self.x = verifier.new_vars(self.n)
        self.x_map = {str(x): x for x in self.x}

        self.f, self.f_whole_domain, self.S_d = system(functions=verifier.solver_fncts())

        self.verifier = verifier(self.n, self.eq, self.inner, self.outer, self.x)
        # self.verifier = verifier(self.n, self.domain, self.initial_s, self.unsafe, vars_bounds, self.x)
        self.domain = self.f_whole_domain(verifier.solver_fncts(), self.x)
        self.S_d = self.S_d(self.x)

        self.xdot = self.f(self.verifier.solver_fncts(), np.array(self.x).reshape(len(self.x), 1))
        self.x = np.matrix(self.x).T
        self.xdot = np.matrix(self.xdot).T

        if learner_type == LearnerType.NN:
            self.learner = self.learner = NN(n_vars, *n_hidden_neurons,
                                             bias=True, activate=activations, equilibria=self.eq)
            self.optimizer = torch.optim.AdamW(self.learner.parameters(), lr=self.learning_rate)
        else:
            raise ValueError('No learner of type {}'.format(learner_type))

        self.f_verifier = partial(self.f, self.verifier.solver_fncts())
        self.f_learner = partial(self.f, self.learner.learner_fncts())

    # the cegis loop
    # todo: fix return, fix map(f, S)
    def solve(self):

        Sdot = self.f_learner(self.S_d.T)
        S, Sdot = self.S_d, torch.stack(Sdot).reshape(self.S_d.shape)

        if self.learner_type == LearnerType.NN:
            self.optimizer = torch.optim.AdamW(self.learner.parameters(), lr=self.learning_rate)

        stats = {}
        # the CEGIS loop
        iters = 0
        stop, found = False, False
        start = timeit.default_timer()
        #
        while not stop:

            print_section('Learning', iters)
            learned = self.learner.learn(self.optimizer, S, Sdot, self.fcts)

            # to disable rounded numbers, set rounding=-1
            if self.sp_handle:
                x_sp = [sp.Symbol('x%d' % i) for i in range(len(self.x))]
                V_s, Vdot_s = get_symbolic_formula(self.learner, sp.Matrix(x_sp),
                                                   self.f_verifier(np.array(x_sp).reshape(len(x_sp), 1)),
                                                   self.eq, rounding=3, lf=self.fcts)
                V_s, Vdot_s = sp.simplify(V_s), sp.simplify(Vdot_s)
                V = sympy_converter(V_s, var_map=self.x_map, target=type(self.verifier))
                Vdot = sympy_converter(Vdot_s, var_map=self.x_map, target=type(self.verifier))
            else: # verifier handles
                V, Vdot = get_symbolic_formula(self.learner, self.x, self.xdot,
                                                   self.eq, rounding=3, lf=self.fcts)
            if self.verifier == Z3Verifier:
                V, Vdot = z3.simplify(V), z3.simplify(Vdot)

            print_section('Candidate', iters)
            print(f'V: {V}')
            print(f'Vdot: {Vdot}')

            print_section('Verification', iters)
            found, ces = self.verifier.verify(V, Vdot)

            if self.max_cegis_iter == iters:
                print('Out of Cegis loops')
                stop = True

            if found:
                print('Found a Lyapunov function, baby!')
                stop = True
            else:
                iters += 1
                if len(ces) > 0:
                    S, Sdot = self.add_ces_to_data(S, Sdot, ces)
                    # the original ctx is in the last row of ces
                    trajectory = self.trajectoriser(ces[-1])
                    S, Sdot = self.add_ces_to_data(S, Sdot, trajectory)

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
        S = torch.cat([S, ces], dim=0)
        Sdot = torch.cat([Sdot, torch.stack(self.f_learner(ces.T)).reshape(ces.shape)], dim=0)
        return S, Sdot

    # NOTA: using ReLU activations, the gradient is often zero
    def trajectoriser(self, point):
        """
        :param point: tensor
        :return: tensor (points towards max Vdot)
        """
        point.requires_grad = True
        trajectory = compute_trajectory(self.learner, point, self.f_learner)

        return torch.stack(trajectory)
