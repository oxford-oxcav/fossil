import torch
import numpy as np
import sympy as sp
import timeit
from z3 import *
import logging

from src.lyap.verifier.verifier import Verifier
from src.shared.consts import LearnerType, VerifierType
from src.lyap.verifier.z3verifier import Z3Verifier
from src.lyap.verifier.drealverifier import DRealVerifier
from src.shared.Trajectoriser import Trajectoriser
from src.lyap.utils import get_symbolic_formula, print_section, compute_trajectory
from src.lyap.learner.net import NN
from functools import partial
from src.shared.sympy_converter import sympy_converter
try:
    import dreal as dr
except Exception as e:
    logging.exception('Exception while importing dReal')


class Cegis:
    # todo: set params for NN and avoid useless definitions
    def __init__(self, n_vars, system, learner_type, activations, n_hidden_neurons,
                 verifier_type, inner_radius, outer_radius,
                 **kw):
        self.sp_simplify = kw.get('sp_simplify', True)
        self.sp_handle = kw.get('sp_handle', True)
        self.fcts = kw.get('factors', None)
        self.eq = kw.get('eq', np.zeros((1, n_vars))) # default equilibrium in zero

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

        self.f, self.f_whole_domain, self.S_d = \
            system(functions=verifier.solver_fncts(), inner=inner_radius, outer=outer_radius)
        # self.S_d = self.S_d.requires_grad_(True)

        # self.verifier = verifier(self.n, self.domain, self.initial_s, self.unsafe, vars_bounds, self.x)
        self.domain = self.f_whole_domain(verifier.solver_fncts(), self.x)
        self.verifier = verifier(self.n, self.eq, self.domain, self.x)

        self.xdot = self.f(self.verifier.solver_fncts(), np.array(self.x).reshape(len(self.x), 1))
        self.x = np.matrix(self.x).T
        self.xdot = np.matrix(self.xdot).T

        if learner_type == LearnerType.NN:
            self.learner = NN(n_vars, *n_hidden_neurons, bias=False, activate=activations, equilibria=self.eq)
            self.optimizer = torch.optim.AdamW(self.learner.parameters(), lr=self.learning_rate)
        else:
            raise ValueError('No learner of type {}'.format(learner_type))

        self.f_verifier = partial(self.f, self.verifier.solver_fncts())
        self.f_learner = partial(self.f, self.learner.learner_fncts())

        self.trajectoriser = Trajectoriser(self.f_learner)

    # the cegis loop
    # todo: fix return, fix map(f, S)
    def solve(self):

        Sdot = self.f_learner(self.S_d.T)
        S, Sdot = self.S_d, torch.stack(Sdot).T

        if self.learner_type == LearnerType.NN:
            self.optimizer = torch.optim.AdamW(self.learner.parameters(), lr=self.learning_rate)

        stats = {}
        # the CEGIS loop
        iters = 0
        stop = False
        learner_to_next_component_inputs = {
            'x_map': self.x_map,
            'x': self.x,
            'xdot': self.xdot,
            'sp_simplify': self.sp_simplify,
            'sp_handle': self.sp_handle,
            'factors': self.fcts,
            'f_verifier': self.f_verifier,
            'eq': self.eq,
        }
        components = [
            {
                'name': 'learner',
                'instance': self.learner,
                'to_next_component': lambda _outputs, next_component, **kw:
                    self.learner.to_next_component(self.learner, next_component, **{
                        **kw, **learner_to_next_component_inputs,
                    }),
            },
            {
                'name': 'verifier',
                'instance': self.verifier,
                'to_next_component': lambda _outputs, next_component, **kw: kw,
            },
        ]

        state = {
            'optimizer': self.optimizer,
            'S': S,
            'Sdot': Sdot,
            'factors': self.fcts,
            'V': None,
            'Vdot': None,
            'found': False,
            'ces': None
        }

        while not stop:
            for component_idx in range(len(components)):
                component = components[component_idx]
                next_component = components[(component_idx + 1) % len(components)]

                print_section(component['name'], iters)
                outputs = component['instance'].get(**state)

                state = {**state, **outputs}

                print_section('Outputs', state['Vdot'])

                state = {**state, **(component['to_next_component'](outputs, next_component['instance'], **state))}

                if state['found']:
                    print('Found a Lyapunov function')
                    stop = True

            if self.max_cegis_iter == iters:
                print('Out of Cegis loops')
                stop = True

            iters += 1
            if not state['found']:
                ces = state['ces']
                if len(ces) > 0:
                    S, Sdot = self.add_ces_to_data(S, Sdot, ces)
                    # the original ctx is in the last row of ces
                    trajectory = self.trajectoriser.compute_trajectory(self.learner, ces[-1])
                    state['S'], state['Sdot'] = \
                        self.add_ces_to_data(state['S'], state['Sdot'], torch.stack(trajectory))

        print('Learner times: {}'.format(self.learner.get_timer()))
        print('Verifier times: {}'.format(self.verifier.get_timer()))
        return self.learner, state['found'], iters

    def add_ces_to_data(self, S, Sdot, ces):
        """
        :param S: torch tensor
        :param Sdot: torch tensor
        :param ces: list of ctx
        :return:
                S: torch tensor, added new ctx
                Sdot torch tensor, added  f(new_ctx)
        """
        S = torch.cat([S, ces], dim=0).detach()
        Sdot = torch.stack(self.f_learner(S.T)).T
        # torch.cat([Sdot, torch.stack(self.f_learner(ces.T)).T], dim=0)
        return S, Sdot

    # NOTA: using ReLU activations, the gradient is often zero
    def trajectoriser_method(self, point):
        """
        :param point: tensor
        :return: tensor (points towards max Vdot)
        """
        point.requires_grad = True
        trajectory = compute_trajectory(self.learner, point, self.f_learner)

        return torch.stack(trajectory)
