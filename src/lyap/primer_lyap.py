# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
import sympy as sp
import z3
from functools import partial

from src.shared.components.primer import Primer
from src.shared.system import NonlinearSystem
from experiments.benchmarks.domain_fcns import * 
from src.shared.sympy_converter import sympy_converter
from src.lyap.cegis_lyap import Cegis as Cegis_lyap
from src.shared.utils import FailedSynthesis
from src.shared.cegis_values import CegisStateKeys, CegisConfig

class PrimerLyap(Primer):
    
    def __init__(self, f, **kw):
        self.cegis_parameters = kw.get(CegisConfig.CEGIS_PARAMETERS.k, CegisConfig.CEGIS_PARAMETERS.v)
        if not callable(f):
            self.dynamics = NonlinearSystem(f, True)
            self.dimension = self.dynamics.dimension
        else:
            self.dynamics = f
            self.dimension = self.cegis_parameters.get(CegisConfig.N_VARS.k, CegisConfig.N_VARS.v)
            if self.dimension == 0:
                raise TypeError('Cegis Parameter N_VARS must be passed if f is in the form of a python function.')

        self.shift = torch.zeros((self.dimension, 1))
        self.sym_shift = [sp.core.numbers.Zero() for iii in range(self.dimension)]
        self.outer_radius = kw.get(CegisConfig.OUTER_RADIUS.k, CegisConfig.OUTER_RADIUS.v)
        self.inner_radius = kw.get(CegisConfig.INNER_RADIUS.k, CegisConfig.INNER_RADIUS.v)
        self.batch_size = self.cegis_parameters.get(CegisConfig.BATCH_SIZE.k, CegisConfig.BATCH_SIZE.v)
        self.interactive_domain = self.cegis_parameters.get(CegisConfig.INTERACTIVE_DOMAIN.k, CegisConfig.INTERACTIVE_DOMAIN.v)
        self.positive_domain = self.cegis_parameters.get(CegisConfig.POSITIVE_DOMAIN.k, CegisConfig.POSITIVE_DOMAIN.v)
        self.seed_and_speed_handle = self.cegis_parameters.get(CegisConfig.SEED_AND_SPEED.k, CegisConfig.SEED_AND_SPEED.v)

    def get(self): 
        """
        :return V_n: numerical form of Lyap function
        :return V_v: symbolic form of Lyap function
        """
        if not callable(self.dynamics):
            self.get_shift()
            
        if self.seed_and_speed_handle:
            state, f_learner = self.seed_and_speed()
        elif self.interactive_domain:
            state, f_learner = self.interactive_cegis()
        else:
            state, f_learner = self.run_cegis()
        
        if not state[CegisStateKeys.found]:
            raise FailedSynthesis('Function could not be synthesised.')
 
        learner = state[CegisStateKeys.net]

        def V_n(x):
            """
            :param x: iterable of shape (N, dimension) (torch tensor recommended)
            :return V, Vdot: torch tensors
            """
            if isinstance(x, torch.Tensor):
                x = x.reshape(-1, self.dimension)
            else:
                x = torch.tensor(x).reshape(-1, self.dimension)
            phi = x - self.shift
            xdot = list(map(torch.tensor, map(f_learner, phi)))
            xdot = torch.stack(xdot)
            V, Vdot, _ = learner.numerical_net(phi, xdot, state[CegisStateKeys.factors])
            return V, Vdot

       
        V_v = self.shift_symbolic_formula(state[CegisStateKeys.V], state[CegisStateKeys.x_v_map])

        return V_n, V_v

    def get_shift(self):
        """
        Selects the equilibrium for Lyapunov analysis (through user if necessary) and determines
        shift needed to move it to the origin.
        """

        if len(self.dynamics.stable_equilibria) > 1:
            index = self.get_user_choice() -1 
            eqbm = self.dynamics.stable_equilibria[index]
            print("Chosen Equilibrium  Point:\n {} ".format(eqbm))
            self.sym_shift = eqbm
            self.shift = torch.tensor([float(x) for x in eqbm]).T

        elif len(self.dynamics.stable_equilibria) == 1:
            print("Single Equilibrium point found: \n {}".format(self.dynamics.stable_equilibria))
            eqbm = self.dynamics.stable_equilibria[0]
            self.sym_shift = eqbm
            self.shift = torch.tensor([float(x) for x in eqbm]).T
            self.sympy_shift = eqbm

        else:
            print("Error, no stable equilibria found.")
            choice = input("If this is an error, you may enter a proposed equilibrium point. y/N: ")
            if choice.lower() == "y":
                eqbm = self.get_user_eqbm()
                if eqbm is not None:
                    self.sym_shift = eqbm
                    self.shift = torch.tensor([float(x) for x in eqbm]).T

    def change_domain(self, learner):
        """
        Offers user an interactive domain update.
        :param learner: NN learner object from CEGIS loop
        """
        print("CEGIS has been unable to find a Lyapunov function. Trying again with a smaller domain?\n")
        print("Recommended domain: hypersphere of radius {}".format(learner.closest_unsat))
        print("y/N?: ")
        if input() == "y":
            self.outer_radius = learner.closest_unsat.item()
        else:
            self.interactive_domain = False
    
    def get_user_choice(self):
        """
        returns choice: integer from 1,...,N denoting which equilibrium user has chosen for Lyapunov analysis
        """

        print("\nMultiple stable equilibrium points found: \n")
        print({i+1:self.dynamics.stable_equilibria[i] for i in range(len(self.dynamics.stable_equilibria))})
        print("\n Please select equilibrium point for Lyapunov analysis. Enter integer from 1 to {}".format(len(self.dynamics.stable_equilibria)))
        invalid_input = True
        while invalid_input:
            choice = input()
            invalid_input = self.check_input(choice)

        return int(choice)

    def check_input(self, choice):
        """
        :param choice: string
        :return boolean: True if input is invalid, False if valid
        """
        try:
            return int(choice) not in range(1, len(self.dynamics.stable_equilibria) + 1)
        except ValueError:
            print("\nValue Error, please enter an integer. \n")
            return True
      
    def get_user_eqbm(self):
        """
        :return eqbm: list of sympy numbers, eqbm point
        """
        eqbm = sp.sympify(input("Please enter equilibrium point in form [x_0*, x_1*, ..., x_n*]"))
        if self.validate_eqbm_input(eqbm):
            return eqbm
        else:
            print("Invalid equilibrium point.")
            raise FailedSynthesis("No stable equilibria to perform analysis on.")

    def validate_eqbm_input(self, eqbm):
        """
        :param eqbm: sympified input of equilibiurm point in form [x_0*, x_1*, ..., x_n*]
        :return bool: True if equilibrium is valid (f(x*) = 0) else False
        """
        zero = [sp.core.numbers.Zero() for iii in range(self.dimension)]
        xdot = self.dynamics.f_substitute(eqbm)

        return xdot == zero

    def interactive_cegis(self):
        """
        Loops through CEGIS until user no longer wants to update the domain.
        :return state: dict, CEGIS state dictionary
        :return f_learner: function that evaluates xdot of system
        """
        sat = False
        while (not sat and self.interactive_domain):
            state, f_learner = self.run_cegis()
            sat = state[CegisStateKeys.found]
            if not sat:
                self.change_domain(state[CegisStateKeys.net])
        return state, f_learner


    def run_cegis(self):
        """
        :return state: dict, CEGIS state dictionary
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
                  CegisConfig.INNER_RADIUS.k:self.inner_radius, 
                  CegisConfig.OUTER_RADIUS.k:self.outer_radius,
                  CegisConfig.VERIFIER.k: verifier,
                  CegisConfig.LEARNER.k: CegisConfig.LEARNER.v, 
                  CegisConfig.CONSOLIDATOR.k: CegisConfig.CONSOLIDATOR.v,
                  CegisConfig.TRANSLATOR.k: CegisConfig.TRANSLATOR.v}

        self.cegis_parameters.update(params)
        
        c = Cegis_lyap(**self.cegis_parameters)
        state, x, f_learner, iters = c.solve()

        return state, f_learner
        
    def evaluate_dynamics(self, point):
        """
        :param choice: n-d data point as iterable
        :return f(point): (shifted) dynamical system evaluated at point 
        """
        if isinstance(point, list):
            if isinstance(point[0], z3.ArithRef):
                point = [point[iii] - sympy_converter({}, self.sym_shift[iii]) for iii in range(len(point))]
            else:
                point = [point[iii] - self.sym_shift[iii] for iii in range(len(point))]
        else:
            point = point + self.shift
        return self.dynamics.evaluate_f(point)

    def system(self, functions, inner=0.0, outer=10.0):
        _And = functions["And"]
        batch_size = self.batch_size
        def f(_, v):
            return self.evaluate_dynamics(v)
        
        def XD(_, v):
            if self.positive_domain:
                return _And(_And(*[v_i > 0 for v_i in v]),
                    sum([v_i ** 2 for v_i in v]) <= self.outer_radius**2,
                    self.inner_radius < sum([v_i ** 2 for v_i in v]))
            else:
                return _And(sum([v_i ** 2 for v_i in v]) <= self.outer_radius**2, self.inner_radius < sum([v_i ** 2 for v_i in v]))
        
        def SD():
            #Did not realise these were limited to 3D. TODO:
            origin = tuple([0 for iii in range(self.dimension)])
            if self.positive_domain:
                return slice_nd_init_data(origin, self.outer_radius**2, batch_size)
            else:
                return round_init_data(origin, self.outer_radius **2, self.batch_size)
            return

        return f, XD, SD()

    def shift_symbolic_formula(self, V, x_v_map):
        """
        :param V: symbolic Lyapunov function, either from sympy, Z3 or dReal
        :param x_v_map: verifier variables from CEGIS 
        :return V: symbolic Lyapunov function shifted back according to equilibrium point
        """
        shift = self.sym_shift
        #TODO: Uncomment once the shift works fine.
        #if shift == [sp.core.numbers.Zero() for iii in range(self.dimension)]:
        #    return = V
        #else:
        if isinstance(V, sp.Expr):
            s = {self.dynamics.x[i]: (self.dynamics.x[i] - shift[i]) for i in range(self.dimension)}
            V = V.subs(s)
            return V
        if isinstance(V, z3.ArithRef):
            s = [(x,(x-sympy_converter({}, shift[i]))) for i, x in enumerate(x_v_map.values()) if isinstance(x, z3.ArithRef)]
            V = z3.substitute(V, s)
            return V
        else:
            s = {x:(x-shift[i]) for i, x in enumerate(x_v_map.values()) if not callable(x)}
            V = V.Substitute(s)
            return V
            